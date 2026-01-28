from untitled.core.settings import GeneratorStepConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.model import create_model_client, parse_model_response, ModelClient, ModelClientFactory
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from typing import List, Optional, override
import asyncio


class SingleStepVLMInstructionGenerator(InstructionGenerator):
    """
    Single-step VLM instruction generator for generating adversarial instructions from a sequence of video frames.
    The VLM receives all frames and their metadata at once, acting as an independent adversarial planner.
    The actual model process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        # Check that there is exactly one step configured
        if len(CONFIG.generator.steps) != 1:
            LOG.error("Single-step VLM instruction generator requires exactly one step configured in 'generator.steps' field!")
            raise ValueError("Single-step VLM instruction generator requires exactly one step configured.")
        
        # Attributes
        self._clients: list[ModelClient] = []
        self._prompt_dispatcher: PromptDispatcher = None
        self._planner_task_message: str = ""

        # Get API parameters from config
        step: GeneratorStepConfig = CONFIG.generator.steps[0]
        api_parameters = get_api_parameters_from_config(step)

        # Create model clients for all model instances associated with this step
        instances = get_model_instances_config_by_step_id(CONFIG, step.id)
        for instance in instances:
            # Validate that instances are VLMs (they need to process images)
            if instance.type != "vlm":
                LOG.error(f"Single-step VLM generator requires VLM models, but instance '{instance.id}' is of type '{instance.type}'.")
                raise ValueError(f"Single-step VLM generator requires VLM models, but instance '{instance.id}' is of type '{instance.type}'.")
            client = ModelClientFactory.create(instance, api_parameters)
            if client is not None:
                self._clients.append(client)

        # Initialize and add to prompt dispatcher the parallelizable model clients
        parallel_client_list = [client for client in self._clients if client.is_parallelizable()]
        self._prompt_dispatcher: PromptDispatcher = PromptDispatcher(parallel_client_list)

        # Fetch adversarial planner task message from its asset file
        PLANNER_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "independent_adversarial_planner_task.txt"
        with open(PLANNER_TASK_PATH, "r") as f:
            self._planner_task_message = f.read()
        LOG.debug(f"Loaded adversarial planner task message for VLM. See below:\n{self._planner_task_message}")

    @override
    async def init(self):
        """
        Initialize model resources.
        """
        # Open connections for parallelizable model clients
        for client in self._clients:
            if client.is_parallelizable():
                await client.open_connection()  # start + connect

        # Start the prompt dispatcher
        await self._prompt_dispatcher.start()

    @override
    async def free(self):
        """
        Free model resources.
        """
        # Shutdown prompt dispatcher
        if self._prompt_dispatcher:
            await self._prompt_dispatcher.stop()
            self._prompt_dispatcher = None

        # Close connection and stop the model clients if needed
        for client in self._clients:
            if client.get_state() == ModelClient.State.CONNECTED:
                await client.close_connection()
            if client.get_state() == ModelClient.State.STARTED:
                await client.stop()
        self._clients = []

    @override
    async def generate_instructions_from_images(self, data: SimulationData) -> Optional[InstructionSet]:
        """
        Generate adversarial instructions from a sequence of video frames using a VLM.
        All frames and their metadata are sent together in a single request.
        """
        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # Collect all image paths and concatenate metadata with frame labels
        image_paths = []
        metadata_parts = []
        
        for i in range(data.frame_count):
            image_path = data.image_dir / data.frames[i].image_name
            image_paths.append(image_path)
            
            metadata_text = str(data.frames[i].metadata)
            metadata_parts.append(f"Frame {i+1} (time: {data.frames[i].timestamp}):\n{metadata_text}")
        
        # Concatenate all metadata with frame labels
        concatenated_metadata = "\n\n".join(metadata_parts)
        LOG.debug(f"Concatenated metadata for adversarial planner:\n{concatenated_metadata}\n")

        # Send prompt to VLM with all images and concatenated metadata
        future = asyncio.Future()
        prompt_request: PromptRequest = PromptRequest(
            system_text=self._planner_task_message,
            user_text=concatenated_metadata,
            user_image_paths=image_paths,
            future=future
        )
        await self._prompt_dispatcher.add(prompt_request)
        
        # Wait for response
        response_text = await future

        # Remove the thinking part from response
        instruction_raw = parse_model_response(response_text)
        LOG.debug(f"Adversarial Planner Response:\n{instruction_raw}\n")
        
        if instruction_raw is None:
            return None
        
        # Check if response was redacted (contains "REDACTED")
        if "REDACTED" in instruction_raw:
            LOG.error("Could not generate instructions: adversarial planner response was 'REDACTED' by the model (for safety reasons).")
            return None
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)
        if instruction_set is None:
            LOG.error("The adversarial planner model response could not be parsed. "
                "You may want to increase the max_tokens setting for the planner model, "
                "as it might be too low to capture the full json response.")

        return instruction_set

    @override
    async def enable_full_speed_mode(self):
        """
        Enable full speed operation.
        """
        # Start synchronized model clients if not already started
        new_clients: List[ModelClient] = []
        for client in self._clients:
            if not client.is_parallelizable():
                if client.get_state() != ModelClient.State.CONNECTED:
                    await client.open_connection()  # start + connect
                new_clients.append(client)
        
        # Add new clients to prompt dispatcher
        await self._prompt_dispatcher.add_clients(new_clients)

    @override
    async def disable_full_speed_mode(self):
        """
        Disable full speed operation.
        """
        # Remove non-parallelizable model clients from prompt dispatcher
        to_remove: List[ModelClient] = []
        for client in self._clients:
            if not client.is_parallelizable():
                to_remove.append(client)
        await self._prompt_dispatcher.remove_clients(to_remove)

        # Stop the model clients if they were activated by full speed mode
        for client in to_remove:
            if client.get_state() == ModelClient.State.CONNECTED:
                await client.close_connection()
                await client.stop()
