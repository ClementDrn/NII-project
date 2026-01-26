import asyncio
from untitled.core.settings import GeneratorStepConfig, ModelInstanceConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.model import create_model_client, parse_model_response, ModelClient, ModelClientFactory, ModelType
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from untitled.core.base import VLM_LOG, PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from typing import List, Optional, override


class MultiStepVLMInstructionGenerator(InstructionGenerator):
    """
    Multi-step instruction generator using VLM for scene description and LLM/VLM for planning.
    The actual model process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(self):
        # Check that there are exactly two steps configured
        if len(CONFIG.generator.steps) != 2:
            LOG.error("Multi-step VLM instruction generator requires exactly two steps configured in 'generator.steps' field!")
            raise ValueError("Multi-step VLM instruction generator requires exactly two steps configured.")
        
        # Attributes
        self._descriptor_clients: list[ModelClient] = []
        self._planner_clients: list[ModelClient] = []
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

        # Create model clients for all model instances associated with descriptor step
        descriptor_instances: List[ModelInstanceConfig] = get_model_instances_config_by_step_id(CONFIG, descriptor_step.id)
        for instance in descriptor_instances:
            # Validate that descriptor instances are VLMs (they need to process images)
            if instance.type != "vlm":
                LOG.error(f"Descriptor step requires VLM models, but instance '{instance.id}' is of type '{instance.type}'.")
                raise ValueError(f"Descriptor step requires VLM models, but instance '{instance.id}' is of type '{instance.type}'.")
            client = ModelClientFactory.create(instance, descriptor_api_parameters)
            if client is not None:
                self._descriptor_clients.append(client)

        # Create model clients for all model instances associated with planner step
        planner_instances = get_model_instances_config_by_step_id(CONFIG, planner_step.id)
        for instance in planner_instances:
            client = ModelClientFactory.create(instance, planner_api_parameters)
            if client is not None:
                self._planner_clients.append(client)

        # Initialize prompt dispatchers with parallelizable model clients
        # Note: Actual connection happens in init() since open_connection() is async
        parallel_descriptor_list = [client for client in self._descriptor_clients if client.is_parallelizable()]
        self._descriptor_dispatcher: PromptDispatcher = PromptDispatcher(parallel_descriptor_list)
        parallel_planner_list = [client for client in self._planner_clients if client.is_parallelizable()]
        self._planner_dispatcher: PromptDispatcher = PromptDispatcher(parallel_planner_list)

        # Fetch scene descriptor task message from its asset file
        SCENE_DESCRIPTOR_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "scene_descriptor_task.txt"
        with open(SCENE_DESCRIPTOR_TASK_PATH, "r") as f:
            self._scene_descriptor_task_message = f.read()
        VLM_LOG.debug(f"Loaded scene descriptor task message for VLM. See below:\n{self._scene_descriptor_task_message}")

        # Fetch adversarial planner task message from its asset file
        PLANNER_TASK_PATH = PROJECT_ROOT / "assets" / "prompts" / "adversarial_planner_task.txt"
        with open(PLANNER_TASK_PATH, "r") as f:
            self._planner_task_message = f.read()
        VLM_LOG.debug(f"Loaded adversarial planner task message. See below:\n{self._planner_task_message}")

    @override
    async def init(self):
        """
        Initialize model resources.
        """
        # Open connections for parallelizable descriptor clients
        for client in self._descriptor_clients:
            if client.is_parallelizable():
                await client.open_connection()  # start + connect

        # Same for parallelizable planner clients
        for client in self._planner_clients:
            if client.is_parallelizable():
                await client.open_connection()  # start + connect

        # Start both prompt dispatchers
        await self._descriptor_dispatcher.start()
        await self._planner_dispatcher.start()

    @override
    async def free(self):
        """
        Free model resources.
        """
        # Shutdown prompt dispatchers
        if self._descriptor_dispatcher:
            await self._descriptor_dispatcher.stop()
            self._descriptor_dispatcher = None
        if self._planner_dispatcher:
            await self._planner_dispatcher.stop()
            self._planner_dispatcher = None

        # Close connection and stop the model clients if needed
        for client in self._descriptor_clients:
            if client.get_state() == ModelClient.State.CONNECTED:
                await client.close_connection()
            if client.get_state() == ModelClient.State.STARTED:
                await client.stop()
        self._descriptor_clients = []

        for client in self._planner_clients:
            if client.get_state() == ModelClient.State.CONNECTED:
                await client.close_connection()
            if client.get_state() == ModelClient.State.STARTED:
                await client.stop()
        self._planner_clients = []

    @override
    async def generate_instructions_from_images(self, data: SimulationData) -> Optional[InstructionSet]:
        """
        Generate instruction from images and metadata using LLM/VLM models.
        There are two steps:
        1. Scene Descriptor: Generate a scene description for each provided frame (VLM required).
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
            # NOTE: Thanks to the first step, if a same local model process is shared, it will not be stopped
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
            scene_description = parse_model_response(chat_response)
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
            # NOTE: Thanks to the first step, if a same local model process is shared, it will not be stopped
            await self._deactivate_clients(self._descriptor_dispatcher, self._descriptor_clients)

            # Finally, activate planner clients
            await self._activate_clients(self._planner_dispatcher, self._planner_clients)

        # The prompt is sent to the dispatcher which will select an available planner model client
        planner_future = asyncio.Future()
        prompt_request: PromptRequest = PromptRequest(
            system_text=self._planner_task_message,
            user_text=concatenated_descriptions,
            user_image_paths=None,
            future=planner_future
        )
        await self._planner_dispatcher.add(prompt_request)
        chat_response = await planner_future
        instruction_raw = parse_model_response(chat_response)
        VLM_LOG.debug(f"- Adversarial Planner Response:\n{instruction_raw}\n")
        
        if instruction_raw is None:
            return None
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)
        if instruction_set is None:
            LOG.error("The adversarial planner model response could not be parsed. "
                "You may want to increase the max_tokens setting for the planner model, "
                "as it might be too low to capture the full json response.")

        return instruction_set

    async def _prepare_clients_activation(self, clients: List[ModelClient]):
        """
        Prepare non-parallelizable model clients for activation.
        This method signals the clients that they will be needed soon.
        """
        for client in clients:
            if not client.is_parallelizable():
                await client.set_hot()

    async def _activate_clients(self, dispatcher: PromptDispatcher, clients: List[ModelClient]):
        """
        Activate (start + connect) a list of model clients.
        """
        # Start synchronized LLM/VLM clients if not already started
        new_clients: List[ModelClient] = []
        for client in clients:
            if not client.is_parallelizable():
                if client.get_state() != ModelClient.State.CONNECTED:
                    await client.open_connection()  # start + connect
                new_clients.append(client)

        # Add new clients to prompt dispatcher
        await dispatcher.add_clients(new_clients)
        
    async def _deactivate_clients(self, dispatcher: PromptDispatcher, clients: List[ModelClient]):
        """
        Deactivate (disconnect + stop) a list of model clients.
        """
        # Remove non-parallelizable model clients from prompt dispatcher
        to_remove: List[ModelClient] = []
        for client in clients:
            if not client.is_parallelizable():
                to_remove.append(client)
        await dispatcher.remove_clients(to_remove)

        # Stop the model clients if they were activated by full speed mode
        for client in to_remove:
            if client.get_state() == ModelClient.State.CONNECTED:
                await client.close_connection()
                await client.stop()

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
