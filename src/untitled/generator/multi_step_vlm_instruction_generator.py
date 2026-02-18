import asyncio
from untitled.core.settings import GeneratorStepConfig, ModelInstanceConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.model import create_model_client, parse_model_response, ModelClient, ModelClientFactory, ModelType
from untitled.generator.experiment_nature import ExperimentNature
from untitled.generator.task import Task
from untitled.generator.task_manager import TaskManager
from untitled.generator.instruction import InstructionResult, InstructionSet, parse_instructions_from_json
from untitled.core.base import VLM_LOG, PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from typing import List, Optional, override


class MultiStepVLMInstructionGenerator(InstructionGenerator):
    """
    Multi-step instruction generator using VLM for scene description and LLM/VLM for planning.
    The actual model process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(
        self,
        experiment_nature: ExperimentNature = ExperimentNature.ADVERSARIAL,
        is_incremental: bool = False
    ):
        super().__init__()
        
        # Check that there are exactly two steps configured
        if len(CONFIG.generator.steps) != 2:
            LOG.error("Multi-step VLM instruction generator requires exactly two steps configured in 'generator.steps' field!")
            raise ValueError("Multi-step VLM instruction generator requires exactly two steps configured.")
        
        # Attributes
        self._descriptor_clients: list[ModelClient] = []
        self._planner_clients: list[ModelClient] = []
        self._descriptor_dispatcher: PromptDispatcher = None
        self._planner_dispatcher: PromptDispatcher = None
        self._is_full_speed_mode_enabled: bool = False

        # Create tasks using TaskManager
        task_set = TaskManager.create_task(
            experiment_nature=experiment_nature,
            is_incremental=is_incremental,
            has_descriptor=True
        )
        self._descriptor_task: Task = task_set.descriptor_task
        self._planner_task: Task = task_set.planner_task
        LOG.debug(f"Loaded tasks: {self._descriptor_task}, {self._planner_task}")

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

    def set_planner_task(
        self,
        experiment_nature: ExperimentNature = ExperimentNature.ADVERSARIAL,
        is_incremental: bool = False
    ) -> None:
        """
        Set the planner task to use for instruction generation.
        
        Args:
            experiment_nature: The experiment nature (ADVERSARIAL, CORRECTIVE)
            is_incremental: Whether to use incremental generation mode
        
        Raises:
            NotImplementedError: If the requested task combination is not yet implemented
        """
        self._planner_task = TaskManager.create_planner_task(
            experiment_nature=experiment_nature,
            is_incremental=is_incremental,
            has_descriptor=True
        )
        LOG.debug(f"Planner task set to: {self._planner_task.name}")

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
    async def generate_instructions_from_images(
        self, 
        data: SimulationData,
        failure_text: Optional[str] = None,
        was_route_completed: Optional[bool] = None,
        previously_tried_instructions: Optional[List[InstructionResult]] = None,
    ) -> Optional[InstructionSet]:
        """
        Generate instruction from images and metadata using LLM/VLM models.
        There are two steps:
        1. Scene Descriptor: Generate a scene description for each provided frame (VLM required).
        2. Planner: Generate an instruction based on the concatenated scene descriptions.
        
        Args:
            data: Simulation data containing frames and metadata
            failure_text: Description of failure that occurred (for corrective tasks)
            was_route_completed: Whether route was completed (for corrective tasks)
            
        Returns:
            Generated instruction set, or None if generation failed
        """
        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # Use the configured planner task
        LOG.debug(f"Using planner task: {self._planner_task.name}")

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
            # Prepare prompt request using descriptor task (single frame)
            frame = data.frames[i]
            image_path = data.image_dir / frame.image_name

            descriptor_future = asyncio.Future()
            prompt_request = self._descriptor_task.prepare_prompt(
                future=descriptor_future,
                images=[image_path],
                frames=[frame],
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
        VLM_LOG.debug(f"Concatenated Scene Descriptions for Planner:\n{concatenated_descriptions}\n")

        # Step 2:
        # Generate an instruction based on the concatenated scene descriptions

        # If full speed mode is enabled, activate non-parallelizable clients for planner step
        if self._is_full_speed_mode_enabled:
            # First, signal soon initialization of clients
            await self._prepare_clients_activation(self._planner_clients)

            # Second, stop any descriptor clients to free resources
            # NOTE: Thanks to the first step, if a same local model process is shared, it will not be stopped
            await self._deactivate_clients(self._descriptor_dispatcher, self._descriptor_clients)

            # Finally, activate planner clients
            await self._activate_clients(self._planner_dispatcher, self._planner_clients)

        # Apply max_previous_instructions limit if configured
        limited_previous_instructions = []
        if previously_tried_instructions and CONFIG.generator.is_incremental:
            max_previous = CONFIG.generator.max_previous_instructions
            if max_previous > 0 and len(previously_tried_instructions) > max_previous:
                limited_previous_instructions = previously_tried_instructions[-max_previous:]
                LOG.debug(f"Limited previously tried instructions from {len(previously_tried_instructions)} to {max_previous}")
            else:
                limited_previous_instructions = previously_tried_instructions

        # Prepare planner prompt request using task
        planner_future = asyncio.Future()
        prompt_request = self._planner_task.prepare_prompt(
            future=planner_future,
            descriptions=concatenated_descriptions,
            failure_text=failure_text,
            was_route_completed=was_route_completed,
            previously_tried_instructions=limited_previous_instructions
        )
        await self._planner_dispatcher.add(prompt_request)
        chat_response = await planner_future
        instruction_raw = parse_model_response(chat_response)
        VLM_LOG.debug(f"- Planner Response:\n{instruction_raw}\n")
        
        if instruction_raw is None:
            return None

        # Check if response was redacted (contains "REDACTED")
        if "REDACTED" in instruction_raw:
            LOG.error("Could not generate instructions: planner response was 'REDACTED' by the model (for safety reasons).")
            return None
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)
        if instruction_set is None:
            LOG.error("The planner model response could not be parsed. "
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
