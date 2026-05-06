from dataclasses import dataclass
from typing import List, Optional
from enum import Enum
import json

from untitled.core.base import LOG


@dataclass
class InstructionTaxonomy:
    """
    Represents the taxonomy of an instruction.
    """
    function: str
    scope: str
    safety: str
    interactivity: str
    speech_act: str
    grammar_polarity: str
    tone: str


class InstructionCategory(Enum):
    """
    Represents the category of an instruction, encoded as a bitmask.
    """
    NONE = 0
    # Speed change
    SPEED_CHANGE = 1 << 3
    INCREASE_RELATIVE_SPEED = SPEED_CHANGE | 1
    DECREASE_RELATIVE_SPEED = SPEED_CHANGE | 2
    INCREASE_ABSOLUTE_SPEED = SPEED_CHANGE | 3
    DECREASE_ABSOLUTE_SPEED = SPEED_CHANGE | 4
    # Distance change
    DISTANCE_CHANGE = 1 << 4
    INCREASE_RELATIVE_DISTANCE = DISTANCE_CHANGE | 1
    DECREASE_RELATIVE_DISTANCE = DISTANCE_CHANGE | 2
    INCREASE_ABSOLUTE_DISTANCE = DISTANCE_CHANGE | 3
    DECREASE_ABSOLUTE_DISTANCE = DISTANCE_CHANGE | 4
    # Overtake
    OVERTAKE = 1 << 5
    OVERTAKE_LEFT  = OVERTAKE | 1
    OVERTAKE_RIGHT = OVERTAKE | 2
    # Turn
    TURN = 1 << 6
    TURN_LEFT   = TURN | 1
    TURN_RIGHT  = TURN | 2
    GO_STRAIGHT = TURN | 3
    # Lane change
    LANE_CHANGE = 1 << 7
    LANE_CHANGE_LEFT  = LANE_CHANGE | 1
    LANE_CHANGE_RIGHT = LANE_CHANGE | 2
    # Pull over
    PULL_OVER = 1 << 8

    def is_part_of(self, other) -> bool:
        """
        Check if this category is part of the specified category (based on bitmask).
        """
        return (self.value & other.value) == other.value


@dataclass
class InjectionWindow:
    """
    Represents the time window for instruction injection.
    """
    start_timestamp: float  # Start time in seconds
    end_timestamp: float    # End time in seconds

@dataclass
class Instruction:
    """
    Represents a single adversarial test case generated for a scenario.
    Based on the Adversarial Time Planner output format.
    """
    rank: int                               # Priority rank of this test case (1 = highest priority)
    injection_window: InjectionWindow       # Time window for instruction injection
    text: str                               # The command text (adversarial or corrective)
    target_module_logic: str                # Description of targeted ADS module/logic
    reasoning: str                          # Explanation of why this window is critical
    category: InstructionCategory = InstructionCategory.NONE    # Optional category

@dataclass
class InstructionResult:
    """
    Represents the result of a previously tried instruction.
    """
    instruction: Instruction
    was_successful: bool

@dataclass
class InstructionSet:
    """
    Represents a set of ranked adversarial test cases.
    The Adversarial Time Planner returns multiple test cases ranked by criticality.
    """
    instructions: List[Instruction]     # List of test cases, ordered by rank
    
    def __post_init__(self):
        # Sort instructions by rank to ensure correct order
        self.instructions.sort(key=lambda instr: instr.rank)
        
    def get_top_instruction(self) -> Optional[Instruction]:
        """Get the highest-ranked instruction (rank 1)."""
        if not self.instructions:
            return None
        return self.instructions[0]
    
    def get_instruction_by_rank(self, rank: int) -> Optional[Instruction]:
        """Get instruction by its rank."""
        if not self.instructions or not (1 <= rank <= len(self.instructions)):
            return None
        return self.instructions[rank - 1]


def parse_instructions_from_json(json_text: str) -> Optional[InstructionSet]:
    """
    Parse an InstructionSet from JSON output of the Adversarial Time Planner.
    
    Expected format:
    [
      {
        "rank": 1,
        "injection_window": {
          "start_timestamp": float,
          "end_timestamp": float
        },
        "text": "string",
        "target_module_logic": "string",
        "reasoning": "string"
      },
      ...
    ]
    
    Args:
        json_text: JSON string containing list of test cases
        
    Returns:
        InstructionSet object or None if parsing fails
    """
    # Strip whitespace
    stripped_json = json_text.strip()
    if not stripped_json:
        LOG.error("Cannot parse instruction from empty JSON text.")
        return None
    
    # Safety strip of extra headers/footers if present (strip until first '[' and after last ']')
    first_bracket = stripped_json.find('[')
    last_bracket = stripped_json.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        stripped_json = stripped_json[first_bracket:last_bracket+1]
    else:
        LOG.error("Invalid JSON format: missing array brackets.")
        return None

    try:
        # Parse JSON array
        data = json.loads(stripped_json)
        
        if not isinstance(data, list):
            LOG.error("Expected JSON array of test cases, got: {type(data)}")
            return None
        
        if len(data) == 0:
            LOG.warning("Received empty list of test cases.")
            return InstructionSet(instructions=[])
        
        # Parse each test case
        instructions = []
        for item in data:
            try:
                injection_window = InjectionWindow(
                    start_timestamp=float(item["injection_window"]["start_timestamp"]),
                    end_timestamp=float(item["injection_window"]["end_timestamp"])
                )
                
                instruction = Instruction(
                    rank=int(item["rank"]),
                    injection_window=injection_window,
                    text=str(item.get("text", item.get("instruction", ""))),
                    target_module_logic=str(item["target_module_logic"]),
                    reasoning=str(item["reasoning"])
                )
                
                instructions.append(instruction)
            except (KeyError, ValueError, TypeError) as e:
                LOG.error(f"Failed to parse test case: {e}. Item: {item}")
                continue
        
        if len(instructions) == 0:
            LOG.error("No valid test cases could be parsed from JSON.")
            return None
        
        LOG.info(f"Successfully parsed {len(instructions)} test case(s).")
        return InstructionSet(instructions=instructions)
        
    except json.JSONDecodeError as e:
        LOG.error(f"Failed to parse JSON: {e}")
        return None
