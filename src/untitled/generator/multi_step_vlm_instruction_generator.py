import asyncio
from untitled.core.settings import GeneratorStepConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.vlm import create_vlm_client, parse_vlm_response, VLMClient, VLMClientFactory
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from untitled.core.base import VLM_LOG, PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from typing import List, Optional, override


class MultiStepVLMInstructionGenerator(InstructionGenerator):
    """
    Vision-Language Model (VLM) interface for generating instructions from images and metadata.
    The actual VLM process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        # Check that there are exactly two steps configured
        if len(CONFIG.generator.steps) != 2:
            LOG.error("Multi-step VLM instruction generator requires exactly two steps configured in 'generator.steps' field!")
            raise ValueError("Multi-step VLM instruction generator requires exactly two steps configured.")
        
        # Attributes
        self._descriptor_clients: list[VLMClient] = []
        self._planner_clients: list[VLMClient] = []
        self._descriptor_dispatcher: PromptDispatcher = None
        self._planner_dispatcher: PromptDispatcher = None
        self._scene_descriptor_task_message: str = ""
        self._planner_task_message: str = ""
        self._is_full_speed_mode_enabled: bool = False

        # Get API parameters from config for both steps
        descriptor_step: GeneratorStepConfig = CONFIG.generator.steps[0]
        planner_step: GeneratorStepConfig = CONFIG.generator.steps[1]
        descriptor_api_parameters = get_api_parameters_from_config(descriptor_step)
        planner_api_parameters = get_api_parameters_from_config(planner_step)

        # Create VLM clients for all model instances associated with descriptor step
        descriptor_instances = get_model_instances_config_by_step_id(CONFIG, descriptor_step.id)
        for instance in descriptor_instances:
            client = VLMClientFactory.create(instance, descriptor_api_parameters)
            if client is not None:
                self._descriptor_clients.append(client)

        # Create VLM clients for all model instances associated with planner step
        planner_instances = get_model_instances_config_by_step_id(CONFIG, planner_step.id)
        for instance in planner_instances:
            client = VLMClientFactory.create(instance, planner_api_parameters)
            if client is not None:
                self._planner_clients.append(client)

        # Initialize and add to prompt dispatchers the parallelizable VLM clients
        parallel_descriptor_vlm_list = []
        for vlm in self._descriptor_clients:
            if vlm.is_parallelizable():
                vlm.open_connection()   # start + connect
            parallel_descriptor_vlm_list.append(vlm)
        self._descriptor_dispatcher: PromptDispatcher = PromptDispatcher(parallel_descriptor_vlm_list)

        parallel_planner_vlm_list = []
        for vlm in self._planner_clients:
            if vlm.is_parallelizable():
                vlm.open_connection()   # start + connect
            parallel_planner_vlm_list.append(vlm)
        self._planner_dispatcher: PromptDispatcher = PromptDispatcher(parallel_planner_vlm_list)

        # Fetch scene descriptor task message from its asset file
        SCENE_DESCRIPTOR_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "scene_descriptor_task.txt"
        with open(SCENE_DESCRIPTOR_TASK_PATH, "r") as f:
            self._scene_descriptor_task_message = f.read()
        VLM_LOG.debug(f"Loaded scene descriptor task message for VLM. See below:\n{self._scene_descriptor_task_message}")

        # Fetch adversarial planner task message from its asset file
        PLANNER_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "adversarial_planner_task.txt"
        with open(PLANNER_TASK_PATH, "r") as f:
            self._planner_task_message = f.read()
        VLM_LOG.debug(f"Loaded adversarial planner task message for VLM. See below:\n{self._planner_task_message}")

    @override
    async def init(self):
        """
        Initialize VLM resources.
        """
        # Start both prompt dispatchers
        await self._descriptor_dispatcher.start()
        await self._planner_dispatcher.start()

    @override
    async def free(self):
        """
        Free VLM resources.
        """
        # Shutdown prompt dispatchers
        if self._descriptor_dispatcher:
            await self._descriptor_dispatcher.stop()
            self._descriptor_dispatcher = None
        if self._planner_dispatcher:
            await self._planner_dispatcher.stop()
            self._planner_dispatcher = None

        # Close connection and stop the VLM clients if needed
        for vlm in self._descriptor_clients:
            if vlm.get_state() == VLMClient.State.CONNECTED:
                await vlm.close_connection()
            if vlm.get_state() == VLMClient.State.STARTED:
                await vlm.stop()
        self._descriptor_clients = []

        for vlm in self._planner_clients:
            if vlm.get_state() == VLMClient.State.CONNECTED:
                await vlm.close_connection()
            if vlm.get_state() == VLMClient.State.STARTED:
                await vlm.stop()
        self._planner_clients = []

    @override
    async def generate_instructions_from_images(self, data: SimulationData) -> Optional[InstructionSet]:
        """
        Generate instruction from images and metadata using a VLM.
        There are two steps:
        1. Scene Descriptor: Generate a scene description for each provided frame.
        2. Adversarial Planner: Generate an adversarial instruction based on the concatenated scene descriptions.
        """
        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # Step 1:
        # Generate a scene description for each frame
        scene_descriptions = []
        futures = []

        # If full speed mode is enabled, activate non-parallelizable clients for descriptor step
        if self._is_full_speed_mode_enabled:
            # First, signal soon initialization of clients
            await self._prepare_clients_activation(self._descriptor_clients)

            # Second, stop any planner clients to free resources
            # NOTE: Thanks to the first step, if a same local VLM process is shared, it will not be stopped
            await self._deactivate_clients(self._planner_dispatcher, self._planner_clients)

            # Finally, activate descriptor clients
            await self._activate_clients(self._descriptor_dispatcher, self._descriptor_clients)
        
        for i in range(data.frame_count):
            # Prepare prompt inputs
            metadata_text = str(data.frames[i].metadata)
            image_path = data.image_dir / data.frames[i].image_name

            # Send prompt to VLM for scene description (returns future)
            descriptor_future = asyncio.Future()
            prompt_request: PromptRequest = PromptRequest(
                system_text=self._scene_descriptor_task_message,
                user_text=metadata_text,
                user_image_paths=[image_path],
                future=descriptor_future
            )
            await self._descriptor_dispatcher.add(prompt_request)
            futures.append(descriptor_future)

        # Wait for all scene descriptions
        for future in futures:
            chat_response = await future
            scene_description = parse_vlm_response(chat_response)
            scene_descriptions.append(scene_description)
            VLM_LOG.debug(f"- Scene Description Response:\n{scene_description}\n")

        # Concatenate all scene descriptions and add index labels
        concatenated_descriptions = "\n\n".join(
            f"Frame {i+1} (time: {data.frames[i].timestamp}):\n{desc}" 
            for i, desc in enumerate(scene_descriptions)
        )
        VLM_LOG.debug(f"Concatenated Scene Descriptions for Adversarial Planner:\n{concatenated_descriptions}\n")

        # Step 2:
        # Generate an adversarial instruction based on the concatenated scene descriptions

        # If full speed mode is enabled, activate non-parallelizable clients for planner step
        if self._is_full_speed_mode_enabled:
            # First, signal soon initialization of clients
            await self._prepare_clients_activation(self._planner_clients)

            # Second, stop any descriptor clients to free resources
            # NOTE: Thanks to the first step, if a same local VLM process is shared, it will not be stopped
            await self._deactivate_clients(self._descriptor_dispatcher, self._descriptor_clients)

            # Finally, activate planner clients
            await self._activate_clients(self._planner_dispatcher, self._planner_clients)

        # The prompt is sent to the dispatcher which will select an available planner VLM client
        planner_future = asyncio.Future()
        prompt_request: PromptRequest = PromptRequest(
            system_text=self._planner_task_message,
            user_text=concatenated_descriptions,
            user_image_paths=None,
            future=planner_future
        )
        await self._planner_dispatcher.add(prompt_request)
        chat_response = await planner_future
        instruction_raw = parse_vlm_response(chat_response)
        VLM_LOG.debug(f"- Adversarial Planner Response:\n{instruction_raw}\n")
        
        if instruction_raw is None:
            return None
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)
        if instruction_set is None:
            LOG.error("The adversarial planner VLM response could not be parsed. "
                "You may want to increase the max_tokens setting for the planner model, "
                "as it might be too low to capture the full json response.")

        return instruction_set

    async def _prepare_clients_activation(self, clients: List[VLMClient]):
        """
        Prepare non-parallelizable VLM clients for activation.
        This method signals the clients that they will be needed soon.
        """
        for vlm in clients:
            if not vlm.is_parallelizable():
                await vlm.set_hot()

    async def _activate_clients(self, dispatcher: PromptDispatcher, clients: List[VLMClient]):
        """
        Activate (start + connect) a list of VLM clients.
        """
        # Start synchronized LLM/VLM clients if not already started
        new_clients: List[VLMClient] = []
        for vlm in clients:
            if not vlm.is_parallelizable():
                if vlm.get_state() != VLMClient.State.CONNECTED:
                    await vlm.open_connection()  # start + connect
                new_clients.append(vlm)

        # Add new clients to prompt dispatcher
        await dispatcher.add_clients(new_clients)
        
    async def _deactivate_clients(self, dispatcher: PromptDispatcher, clients: List[VLMClient]):
        """
        Deactivate (disconnect + stop) a list of VLM clients.
        """
        # Remove non-parallelizable VLM clients from prompt dispatcher
        to_remove: List[VLMClient] = []
        for vlm in clients:
            if not vlm.is_parallelizable():
                to_remove.append(vlm)
        await dispatcher.remove_clients(to_remove)

        # Stop the VLM clients if they were activated by full speed mode
        for vlm in to_remove:
            if vlm.get_state() == VLMClient.State.CONNECTED:
                await vlm.close_connection()
                await vlm.stop()

    @override
    async def enable_full_speed_mode(self):
        """
        Enable full speed operation.
        """
        # NOTE: Clients will be activated on demand during instruction generation
        self._is_full_speed_mode_enabled = True
        LOG.debug("InstructionGenerator: Full speed mode enabled.")

    @override
    async def disable_full_speed_mode(self):
        """
        Disable full speed operation.
        """
        # Stop and remove non-parallelizable clients from both dispatchers (if still active)
        await self._deactivate_clients(self._descriptor_dispatcher, self._descriptor_clients)
        await self._deactivate_clients(self._planner_dispatcher, self._planner_clients)

        self._is_full_speed_mode_enabled = False
        LOG.debug("InstructionGenerator: Full speed mode disabled.")
