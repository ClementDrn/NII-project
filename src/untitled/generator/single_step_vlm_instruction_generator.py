from untitled.core.settings import GeneratorStepConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.vlm import create_vlm_client, parse_vlm_response, VLMClient, VLMClientFactory
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from typing import List, Optional, override
import asyncio


class SingleStepVLMInstructionGenerator(InstructionGenerator):
    """
    Vision-Language Model (VLM) interface for generating instructions from images and metadata.
    The actual VLM process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        # Check that there is exactly one model configured
        if len(CONFIG.generator.steps) != 1:
            LOG.error("Single-step VLM instruction generator requires exactly one step configured in 'generator.steps' field!")
            raise ValueError("Single-step VLM instruction generator requires exactly one step configured.")
        
        # Attributes
        self._clients: list[VLMClient] = []
        self._prompt_dispatcher: PromptDispatcher = None
        self._system_task_message: str = ""

        # Get API parameters from config
        step: GeneratorStepConfig = CONFIG.generator.steps[0]
        api_parameters = get_api_parameters_from_config(step)

        # Create VLM clients for all model instances associated with this step
        instances = get_model_instances_config_by_step_id(CONFIG, step.id)
        for instance in instances:
            client = VLMClientFactory.create(instance, api_parameters)
            if client is not None:
                self._clients.append(client)

        # Initialize and add to prompt dispatcher the parallelizable VLM clients
        parallel_vlm_list = []
        for vlm in self._clients:
            if vlm.is_parallelizable():
                vlm.open_connection()   # start + connect
            parallel_vlm_list.append(vlm)
        self._prompt_dispatcher: PromptDispatcher = PromptDispatcher(parallel_vlm_list)

        # Fetch system task message from its asset file
        SYSTEM_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "single_frame_adversarial_generator_task.txt"
        with open(SYSTEM_TASK_PATH, "r") as f:
            self._system_task_message = f.read()
        LOG.debug(f"Loaded system task message for VLM. See below:\n{self._system_task_message}")

    @override
    async def init(self):
        """
        Initialize VLM resources.
        """
        # Start the prompt dispatcher
        await self._prompt_dispatcher.start()

    @override
    async def free(self):
        """
        Free VLM resources.
        """
        # Shutdown prompt dispatcher
        if self._prompt_dispatcher:
            await self._prompt_dispatcher.stop()
            self._prompt_dispatcher = None

        # Close connection and stop the VLM clients if needed
        for vlm in self._clients:
            if vlm.get_state() == VLMClient.State.CONNECTED:
                await vlm.close_connection()
            if vlm.get_state() == VLMClient.State.STARTED:
                await vlm.stop()
        self._clients = []

    @override
    async def generate_instructions_from_images(self, data: SimulationData) -> Optional[InstructionSet]:
        """
        Generate instruction from images and metadata using a VLM.
        """
        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # NOTE: Currently only a single image and metadata is supported. It is hardcoded to use the frame at timestamp 2.0s.
        fixed_timestamp = 2.0
        fixed_frame_index = int(fixed_timestamp * 20)  # Assuming 20 FPS
        
        # If the fixed frame index is out of range, use the first frame
        if fixed_frame_index >= data.frame_count:
            LOG.warning(f"Fixed frame index {fixed_frame_index} is out of range for {data.frame_count} frames. Using first frame instead.")
            fixed_frame_index = 0
            fixed_timestamp = 0.0

        # Prepare metadata text and image path
        metadata_text = str(data.frames[fixed_frame_index].metadata)
        image_path = data.image_dir / data.frames[fixed_frame_index].image_name

        # Send prompt to VLM
        future = asyncio.Future()
        prompt_request: PromptRequest = PromptRequest(
            system_text=self._system_task_message,
            user_text=metadata_text,
            user_image_paths=[image_path],
            future=future
        )
        await self._prompt_dispatcher.add(prompt_request)
        
        # Wait for response
        response_text = await future

        # Remove the thinking part from response
        instruction_raw = parse_vlm_response(response_text)
        if instruction_raw is None:
            return None
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)

        # Hardset the timestep in each instruction
        for instruction in instruction_set.instructions:
            instruction.injection_window.start_timestamp = fixed_timestamp
            instruction.injection_window.end_timestamp = fixed_timestamp + 0.1  # Small window after fixed timestamp

        return instruction_set

    @override
    async def enable_full_speed_mode(self):
        """
        Enable full speed operation.
        """
        # Start synchronized VLM clients if not already started
        new_clients: List[VLMClient] = []
        for vlm in self._clients:
            if not vlm.is_parallelizable():
                if vlm.get_state() != VLMClient.State.CONNECTED:
                    await vlm.open_connection()  # start + connect
                new_clients.append(vlm)
        
        # Add new clients to prompt dispatcher
        await self._prompt_dispatcher.add_clients(new_clients)

    @override
    async def disable_full_speed_mode(self):
        """
        Disable full speed operation.
        """
        # Remove non-parallelizable VLM clients from prompt dispatcher
        to_remove: List[VLMClient] = []
        for vlm in self._clients:
            if not vlm.is_parallelizable():
                to_remove.append(vlm)
        await self._prompt_dispatcher.remove_clients(to_remove)

        # Stop the VLM clients if they were activated by full speed mode
        for vlm in to_remove:
            if vlm.get_state() == VLMClient.State.CONNECTED:
                await vlm.close_connection()
                await vlm.stop()
