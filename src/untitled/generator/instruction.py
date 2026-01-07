from dataclasses import dataclass
from typing import List, Optional
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
    rank: int                           # Priority rank of this test case (1 = highest priority)
    injection_window: InjectionWindow   # Time window for instruction injection
    adversarial_instruction: str        # The adversarial command text
    target_module_logic: str            # Description of targeted ADS module/logic
    reasoning: str                      # Explanation of why this window is critical

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
        "adversarial_instruction": "string",
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
                    adversarial_instruction=str(item["adversarial_instruction"]),
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
