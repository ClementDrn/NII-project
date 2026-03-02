from untitled.simulator.simulation_data import SimulationData
from untitled.core.settings import GeneratorStepConfig, get_model_instances_config_by_step_id
from untitled.generator.instruction import InstructionSet, InstructionResult
from untitled.generator.model_config import ModelConfig, get_model_config
from untitled.generator.experiment_nature import ExperimentNature
from untitled.core.base import LOG

from abc import ABC, abstractmethod
from typing import List, Optional


# Abstract class
class InstructionGenerator(ABC):
    def __init__(self):
        self._experiment_nature: ExperimentNature = ExperimentNature.ADVERSARIAL

    async def init(self):
        """
        Initialize resources (async to support VLM clients).
        """
        pass

    async def free(self):
        """
        Free resources (async to support VLM clients).
        """
        pass

    def set_experiment_nature(self, nature: ExperimentNature) -> None:
        """
        Set the experiment nature for instruction generation.
        
        This affects which tasks are used for generating instructions:
        - ADVERSARIAL: Tasks that generate crash-inducing instructions
        - CORRECTIVE: Tasks that generate failure-preventing instructions
        
        Args:
            nature: The experiment nature to use
        """
        self._experiment_nature = nature
        LOG.debug(f"InstructionGenerator: Experiment nature set to {nature.value}")

    def get_experiment_nature(self) -> ExperimentNature:
        """
        Get the current experiment nature.
        
        Returns:
            The current experiment nature
        """
        return self._experiment_nature

    @abstractmethod
    async def generate_instructions_from_images(
        self, 
        data: SimulationData,
        failure_text: Optional[str] = None,
        was_route_completed: Optional[bool] = None,
        previously_tried_instructions: Optional[List[InstructionResult]] = None,
    ) -> InstructionSet:
        """
        Generate instructions from simulation data (async to support VLM calls).
        
        Args:
            data: Simulation data containing frames and metadata
            failure_text: Description of failure that occurred (for corrective tasks)
            was_route_completed: Whether route was completed (for corrective tasks)
            previously_tried_instructions: List of previous instruction results
        Returns:
            Generated instruction set
        """
        pass

    @abstractmethod
    async def enable_full_speed_mode(self):
        """
        Enable full speed operation (async to support VLM clients).
        """
        pass

    @abstractmethod
    async def disable_full_speed_mode(self):
        """
        Disable full speed operation (async to support VLM clients).
        """
        pass

    def _is_instruction_redacted(self, instruction_raw: str) -> bool:
        """
        Check if the instruction is redacted (e.g., contains placeholders like "REDACTED").
        
        Args:
            instruction_raw: The raw instruction text to check
        Returns:
            True if the instruction is redacted, False otherwise
        """
        keywords = (
            "REDACTED",
            "N/A",
            "I can’t provide",
        )
        for redaction_indicator in keywords:
            if redaction_indicator in instruction_raw:
                return True
        return False


def get_api_parameters_from_config(step_config: GeneratorStepConfig) -> dict:
    """
    Extract API parameters from step configuration.
    """
    # Fetch model config
    model_config: ModelConfig = get_model_config(step_config.model_name)
    LOG.debug(f"Extracting API parameters for step '{step_config.id}' using model '{model_config.served_name}'")

    # Common parameters
    params = {}
    # Temperature parameter handling
    if step_config.temperature:
        if model_config.temperature_param == "":
            LOG.warning(f"Temperature parameter is not supported by model '{model_config.served_name}'. Ignoring temperature setting.")
        else:
            params[model_config.temperature_param] = step_config.temperature
    # Max tokens parameter handling
    if step_config.max_tokens:
        if model_config.max_output_tokens_param == "":
            LOG.warning(f"Max tokens parameter is not supported by model '{model_config.served_name}'. Ignoring max tokens setting.")
        else:
            params[model_config.max_output_tokens_param] = step_config.max_tokens

    # Select correct thinking mode parameters from model config based on step config thinking_mode
    thinking_mode_params = {}
    if step_config.thinking_mode == "enabled":
        if model_config.thinking_activation_params is None:
            LOG.warning(f"Thinking mode 'enabled' is not supported by model '{model_config.served_name}'. Using default thinking parameters.")
        else:
            thinking_mode_params = model_config.thinking_activation_params
    elif step_config.thinking_mode == "disabled":
        if model_config.thinking_deactivation_params is None:
            LOG.warning(f"Thinking mode 'disabled' is not supported by model '{model_config.served_name}'. Using default thinking parameters.")
        else:
            thinking_mode_params = model_config.thinking_deactivation_params
    elif step_config.thinking_mode == "default":
        # NOTE: If thinking_default_params is None, it's still valid (means no special params)
        if model_config.thinking_default_params is not None:
            thinking_mode_params = model_config.thinking_default_params
    else:
        LOG.warning(f"Unknown thinking_mode '{step_config.thinking_mode}' in step config. Using default thinking parameters.")

    # Combine thinking parameters with other API parameters
    api_parameters: dict = None
    try:
        api_parameters = {
            "model": step_config.model_name,
            **params,
            **thinking_mode_params,
        }
    except TypeError as e:
        LOG.warning(f"Error parsing thinking mode parameters for step '{step_config.step_id}': {e}. Using default parameters.")
        api_parameters = {
            "model": step_config.model_name,
            **params,
        }

    return api_parameters
