from untitled.generator.instruction import InstructionSet
from abc import ABC, abstractmethod


# Abstract class
class InstructionGenerator(ABC):
    def __init__(self):
        pass

    def free(self):
        pass

    @abstractmethod
    def generate_instructions_from_images(self, data) -> InstructionSet:
        pass
