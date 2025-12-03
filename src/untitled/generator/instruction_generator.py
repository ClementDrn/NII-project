from untitled.generator.instruction import Instruction
from abc import ABC, abstractmethod


# Abstract class
class InstructionGenerator(ABC):
    def __init__(self):
        pass

    def free(self):
        pass

    @abstractmethod
    def generate_instruction_from_images(self, data) -> Instruction:
        pass
