from untitled.simulator.simulation_data import SimulationData
from untitled.core.settings import GeneratorStepConfig, get_model_instances_config_by_step_id
from untitled.generator.instruction import InstructionSet

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


def get_api_parameters_from_config(step: GeneratorStepConfig) -> dict:
    """
    Extract API parameters from step configuration.
    """
    api_parameters = {
        "model": step.model_name,
        "temperature": step.temperature,
        "max_tokens": step.max_tokens,
        "extra_body": step.extra_body,
    }
    return api_parameters
