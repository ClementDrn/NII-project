from untitled.core.settings import GeneratorStepConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.model import create_model_client, parse_model_response, ModelClient, ModelClientFactory
from untitled.generator.experiment_nature import ExperimentNature
from untitled.generator.task import Task
from untitled.generator.task_manager import TaskManager
from untitled.core.base import PROJECT_ROOT, LOG, VLM_LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.instruction import InstructionCategory, InstructionResult, InstructionSet, parse_instructions_from_json
from typing import List, Optional, override
import asyncio


class SingleStepVLMInstructionGenerator(InstructionGenerator):
    """
    Single-step VLM instruction generator for generating instructions from a sequence of video frames.
    The VLM receives all frames and their metadata at once, acting as an independent planner.
    The actual model process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    """
    def __init__(
        self,
        experiment_nature: ExperimentNature = ExperimentNature.ADVERSARIAL,
        is_incremental: bool = False
    ):
        super().__init__()
        
        # Check that there is exactly one step configured
        if len(CONFIG.generator.steps) != 1:
            LOG.error("Single-step VLM instruction generator requires exactly one step configured in 'generator.steps' field!")
            raise ValueError("Single-step VLM instruction generator requires exactly one step configured.")
        
        # Attributes
        self._clients: list[ModelClient] = []
        self._prompt_dispatcher: PromptDispatcher = None
        
        # Create task using TaskManager
        self._task: Task = TaskManager.create_planner_task(
            experiment_nature=experiment_nature,
            is_incremental=is_incremental,
            has_descriptor=False
        )
        LOG.debug(f"Loaded task: {self._task}")

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

    def set_task(
        self,
        experiment_nature: ExperimentNature = ExperimentNature.ADVERSARIAL,
        is_incremental: bool = False
    ) -> None:
        """
        Set the task to use for instruction generation.
        
        Args:
            experiment_nature: The experiment nature (ADVERSARIAL, CORRECTIVE)
            is_incremental: Whether to use incremental generation mode
        
        Raises:
            NotImplementedError: If the requested task combination is not yet implemented
        """
        self._task = TaskManager.create_planner_task(
            experiment_nature=experiment_nature,
            is_incremental=is_incremental,
            has_descriptor=False
        )
        LOG.debug(f"Task set to: {self._task.name}")

    @override
    async def init(self):
        """
        Initialize model resources.
        """
        await super().init()

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

        await super().free()

    @override
    async def generate_instructions(
        self, 
        data: SimulationData,
        failure_text: Optional[str] = None,
        was_route_completed: Optional[bool] = None,
        previously_tried_instructions: Optional[List[InstructionResult]] = None,
        instruction_category: InstructionCategory = InstructionCategory.NONE,
        timestamp: Optional[float] = None,
    ) -> InstructionSet:
        """
        Generate instructions from a sequence of video frames using a VLM.
        All frames and their metadata are sent together in a single request.
        
        Args:
            data: Simulation data containing frames and metadata
            failure_text: Description of failure that occurred (for corrective tasks)
            was_route_completed: Whether route was completed (for corrective tasks)
            previously_tried_instructions: List of previous instruction results (for context-aware generation)
        Returns:
            Generated instruction set, which may be empty if generation failed or was redacted
        """
        # Check that CONFIG.generator is defined
        if CONFIG.generator is None:
            LOG.error("CONFIG.generator is not defined! Please define generator settings in the configuration file.")
            raise ValueError("CONFIG.generator is not defined! Please define generator settings in the configuration file.")

        # Check frame count validity
        if data.frame_count < 1:
            LOG.error("At least one frame is required to generate instructions!")
            raise ValueError("At least one frame is required to generate instructions.")

        # Use the configured task
        LOG.debug(f"Using task: {self._task.name}")

        # Collect all image paths
        image_paths = [data.image_dir / frame.image_name for frame in data.frames]

        # Apply max_previous_instructions limit if configured
        limited_previous_instructions = []
        if previously_tried_instructions and CONFIG.generator.is_incremental:
            max_previous = CONFIG.generator.max_previous_instructions
            if max_previous > 0 and len(previously_tried_instructions) > max_previous:
                limited_previous_instructions = previously_tried_instructions[-max_previous:]
                LOG.debug(f"Limited previously tried instructions from {len(previously_tried_instructions)} to {max_previous}")
            else:
                limited_previous_instructions = previously_tried_instructions

        # Prepare prompt request using task
        future = asyncio.Future()
        prompt_request = self._task.prepare_prompt(
            future=future,
            images=image_paths,
            frames=data.frames,
            failure_text=failure_text,
            was_route_completed=was_route_completed,
            previously_tried_instructions=limited_previous_instructions
        )
        VLM_LOG.debug(f"Prepared prompt for planner:\n{prompt_request.request.user_content}\n")

        # Send prompt to VLM
        await self._prompt_dispatcher.add(prompt_request)
        
        # Wait for response
        response_text = await future

        # Remove the thinking part from response
        instruction_raw = parse_model_response(response_text)
        LOG.debug(f"Planner Response:\n{instruction_raw}\n")
        
        if instruction_raw is None:
            return InstructionSet(instructions=[])
        
        # Check if response was redacted (contains "REDACTED")
        if self._is_instruction_redacted(instruction_raw):
            LOG.error("Could not generate instructions: planner response was 'REDACTED' by the model (for safety reasons).")
            return InstructionSet(instructions=[])
        
        # Parse instruction set from JSON
        instruction_set = parse_instructions_from_json(instruction_raw)
        if instruction_set is None:
            LOG.error("The planner model response could not be parsed. "
                "You may want to increase the max_tokens setting for the planner model, "
                "as it might be too low to capture the full json response. "
                "The response might also be redacted.")
            instruction_set = InstructionSet(instructions=[])

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
