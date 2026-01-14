from untitled.simulator.simulation_data import SimulationData
from untitled.core.settings import GeneratorStepConfig, get_model_instances_config_by_step_id
from untitled.generator.instruction import InstructionSet
from untitled.generator.model_config import ModelConfig, get_model_config
from untitled.core.base import LOG

from abc import ABC, abstractmethod


# Abstract class
class InstructionGenerator(ABC):
    def __init__(self):
        pass

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

    @abstractmethod
    async def generate_instructions_from_images(self, data: SimulationData) -> InstructionSet:
        """
        Generate instructions from simulation data (async to support VLM calls).
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


def get_api_parameters_from_config(step_config: GeneratorStepConfig) -> dict:
    """
    Extract API parameters from step configuration.
    """
    # Fetch model config
    model_config: ModelConfig = get_model_config(step_config.model_name)

    # Select correct thinking mode parameters from model config based on step config thinking_mode
    thinking_mode_params = {}
    if step_config.thinking_mode == "enabled":
        if model_config.thinking_enabled_params is None:
            LOG.warning(f"Thinking mode 'enabled' is not supported by model '{model_config.name}'. Using default thinking parameters.")
        else:
            thinking_mode_params = model_config.thinking_enabled_params
    elif step_config.thinking_mode == "disabled":
        if model_config.thinking_disabled_params is None:
            LOG.warning(f"Thinking mode 'disabled' is not supported by model '{model_config.name}'. Using default thinking parameters.")
        else:
            thinking_mode_params = model_config.thinking_disabled_params
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
            "temperature": step_config.temperature,
            "max_tokens": step_config.max_tokens,
            **thinking_mode_params,
        }
    except TypeError as e:
        LOG.warning(f"Error parsing thinking mode parameters for step '{step_config.step_id}': {e}. Using default parameters.")
        api_parameters = {
            "model": step_config.model_name,
            "temperature": step_config.temperature,
            "max_tokens": step_config.max_tokens,
        }

    return api_parameters
