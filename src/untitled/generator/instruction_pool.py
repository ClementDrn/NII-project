from untitled.generator.instruction import Instruction, InstructionCategory

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class InstructionPoolCategory:
    """
    Augments the InstructionCategory with additional data for instruction generation.

    For example, for speed change instructions, the extra variable could represent the speed delta (e.g., +10 m/s).
    """
    category: InstructionCategory = InstructionCategory.NONE
    variable: int = 0       # e.g., meters for distance, m/s for speed


class InstructionPool:
    """
    Represents a pool of instruction strings for different categories. 
    """
    def __init__(self):
        self._instructions_by_category: Dict[InstructionPoolCategory, List[str]] = {}

    def add_instructions(self, category: InstructionPoolCategory, instructions: List[str]):
        """
        Add a list of instruction strings to the pool under the specified category.
        """
        if category not in self._instructions_by_category:
            self._instructions_by_category[category] = []
        self._instructions_by_category[category].extend(instructions)

    def get_categories(self) -> List[InstructionPoolCategory]:
        """
        Get the list of instruction categories available in the pool.
        """
        return list(self._instructions_by_category.keys())

    def get_instructions_by_category(self, category: InstructionPoolCategory) -> List[str]:
        """
        Get the list of instruction strings for the specified category.
        """
        return self._instructions_by_category.get(category, [])
