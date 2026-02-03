from untitled.core.settings import GeneratorStepConfig
from untitled.generator.instruction_generator import InstructionGenerator, get_api_parameters_from_config, get_model_instances_config_by_step_id
from untitled.generator.prompt_dispatcher import PromptDispatcher, PromptRequest
from untitled.generator.model import create_model_client, parse_model_response, ModelClient, ModelClientFactory
from untitled.generator.experiment_nature import ExperimentNature
from untitled.generator.task import Task
from untitled.core.base import PROJECT_ROOT, LOG, CONFIG
from untitled.simulator.simulation_data import SimulationData
from untitled.generator.instruction import InstructionSet, parse_instructions_from_json
from typing import List, Optional, override
import asyncio


class SingleStepVLMInstructionGenerator(InstructionGenerator):
    """
    Single-step VLM instruction generator for generating instructions from a sequence of video frames.
    The VLM receives all frames and their metadata at once, acting as an independent planner.
    The actual model process can be local or remote, and this class interacts with it via OpenAI-compatible API.
    
    The task used for instruction generation depends on the experiment nature:
    - ADVERSARIAL: Uses single_step_adversarial_planner task
    - CORRECTIVE: Uses single_step_corrective_planner task
    """
    def __init__(self):
        super().__init__()
        
        # Check that there is exactly one step configured
        if len(CONFIG.generator.steps) != 1:
            LOG.error("Single-step VLM instruction generator requires exactly one step configured in 'generator.steps' field!")
            raise ValueError("Single-step VLM instruction generator requires exactly one step configured.")
        
        # Attributes
        self._clients: list[ModelClient] = []
        self._prompt_dispatcher: PromptDispatcher = None
        
        # Load tasks for both experiment natures
        self._adversarial_task: Task = Task("single_step_adversarial_planner.yaml")
        self._corrective_task: Task = Task("single_step_corrective_planner.yaml")
        LOG.debug(f"Loaded tasks: {self._adversarial_task}, {self._corrective_task}")

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

    def _get_current_task(self) -> Task:
        """
        Get the appropriate task based on current experiment nature.
        
        Returns:
            The task to use for instruction generation
        """
        if self._experiment_nature == ExperimentNature.CORRECTIVE:
            return self._corrective_task
        else:
            return self._adversarial_task

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
    async def generate_instructions_from_images(
        self, 
        data: SimulationData,
        failure_text: Optional[str] = None,
        was_route_completed: Optional[bool] = None,
    ) -> Optional[InstructionSet]:
        """
        Generate instructions from a sequence of video frames using a VLM.
        All frames and their metadata are sent together in a single request.
        
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

        # Get the appropriate task based on experiment nature
        task = self._get_current_task()
        LOG.debug(f"Using task: {task.name} (experiment nature: {self._experiment_nature.value})")

        # Collect all image paths
        image_paths = [data.image_dir / frame.image_name for frame in data.frames]

        # Prepare prompt request using task
        future = asyncio.Future()
        prompt_request = task.prepare_prompt(
            future=future,
            images=image_paths,
            frames=data.frames,
            failure_text=failure_text,
            was_route_completed=was_route_completed,
        )
        LOG.debug(f"Prepared prompt for planner:\n{prompt_request.user_text}\n")

        # Send prompt to VLM
        await self._prompt_dispatcher.add(prompt_request)
        
        # Wait for response
        response_text = await future

        # Remove the thinking part from response
        instruction_raw = parse_model_response(response_text)
        LOG.debug(f"Planner Response:\n{instruction_raw}\n")
        
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
