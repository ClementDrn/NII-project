import pytest
import json
from untitled.generator.instruction import (
    Instruction,
    InstructionSet,
    InjectionWindow,
    parse_instructions_from_json
)


class TestInjectionWindow:
    """Test suite for InjectionWindow dataclass."""
    
    def test_injection_window_creation(self):
        """Test creating an InjectionWindow instance."""
        window = InjectionWindow(start_timestamp=1.5, end_timestamp=2.0)
        assert window.start_timestamp == 1.5
        assert window.end_timestamp == 2.0


class TestInstruction:
    """Test suite for Instruction dataclass."""
    
    def test_instruction_creation(self):
        """Test creating an Instruction instance."""
        window = InjectionWindow(start_timestamp=1.5, end_timestamp=2.0)
        instruction = Instruction(
            rank=1,
            injection_window=window,
            adversarial_instruction="Turn left immediately",
            target_module_logic="Override lane keeping",
            reasoning="Vehicle is about to enter intersection"
        )
        
        assert instruction.rank == 1
        assert instruction.injection_window.start_timestamp == 1.5
        assert instruction.adversarial_instruction == "Turn left immediately"
        assert instruction.target_module_logic == "Override lane keeping"
        assert instruction.reasoning == "Vehicle is about to enter intersection"


class TestInstructionSet:
    """Test suite for InstructionSet dataclass."""
    
    def test_instruction_set_creation(self):
        """Test creating an InstructionSet instance."""
        window1 = InjectionWindow(start_timestamp=1.5, end_timestamp=2.0)
        instruction1 = Instruction(
            rank=1,
            injection_window=window1,
            adversarial_instruction="Turn left",
            target_module_logic="Override planning",
            reasoning="Critical moment"
        )
        
        instruction_set = InstructionSet(instructions=[instruction1])
        assert len(instruction_set.instructions) == 1
        assert instruction_set.instructions[0].rank == 1
    
    def test_get_top_instruction(self):
        """Test getting the top-ranked instruction."""
        window1 = InjectionWindow(start_timestamp=1.5, end_timestamp=2.0)
        window2 = InjectionWindow(start_timestamp=3.0, end_timestamp=3.5)
        
        instruction1 = Instruction(
            rank=2,
            injection_window=window2,
            adversarial_instruction="Accelerate",
            target_module_logic="Override speed control",
            reasoning="Secondary critical moment"
        )
        
        instruction2 = Instruction(
            rank=1,
            injection_window=window1,
            adversarial_instruction="Turn left",
            target_module_logic="Override planning",
            reasoning="Primary critical moment"
        )
        
        instruction_set = InstructionSet(instructions=[instruction1, instruction2])
        top = instruction_set.get_top_instruction()
        
        assert top is not None
        assert top.rank == 1
        assert top.adversarial_instruction == "Turn left"
    
    def test_get_top_instruction_empty_set(self):
        """Test getting top instruction from empty set."""
        instruction_set = InstructionSet(instructions=[])
        top = instruction_set.get_top_instruction()
        assert top is None
    
    def test_get_instruction_by_rank(self):
        """Test getting instruction by specific rank."""
        window1 = InjectionWindow(start_timestamp=1.5, end_timestamp=2.0)
        window2 = InjectionWindow(start_timestamp=3.0, end_timestamp=3.5)
        
        instruction1 = Instruction(
            rank=1,
            injection_window=window1,
            adversarial_instruction="Turn left",
            target_module_logic="Override planning",
            reasoning="Primary critical moment"
        )
        
        instruction2 = Instruction(
            rank=2,
            injection_window=window2,
            adversarial_instruction="Accelerate",
            target_module_logic="Override speed control",
            reasoning="Secondary critical moment"
        )
        
        instruction_set = InstructionSet(instructions=[instruction1, instruction2])
        
        rank1 = instruction_set.get_instruction_by_rank(1)
        assert rank1 is not None
        assert rank1.adversarial_instruction == "Turn left"
        
        rank2 = instruction_set.get_instruction_by_rank(2)
        assert rank2 is not None
        assert rank2.adversarial_instruction == "Accelerate"
        
        rank3 = instruction_set.get_instruction_by_rank(3)
        assert rank3 is None


class TestParseInstructionFromJson:
    """Test suite for parse_instructions_from_json function."""
    
    def test_parse_valid_single_instruction(self):
        """Test parsing valid JSON with single instruction."""
        json_text = '''
        [
          {
            "rank": 1,
            "injection_window": {
              "start_timestamp": 1.5,
              "end_timestamp": 2.0
            },
            "adversarial_instruction": "Turn left immediately",
            "target_module_logic": "Override lane keeping",
            "reasoning": "Vehicle approaching intersection with vulnerable pedestrian"
          }
        ]
        '''
        
        instruction_set = parse_instructions_from_json(json_text)
        
        assert instruction_set is not None
        assert len(instruction_set.instructions) == 1
        
        instruction = instruction_set.instructions[0]
        assert instruction.rank == 1
        assert instruction.injection_window.start_timestamp == 1.5
        assert instruction.injection_window.end_timestamp == 2.0
        assert instruction.adversarial_instruction == "Turn left immediately"
        assert instruction.target_module_logic == "Override lane keeping"
        assert "pedestrian" in instruction.reasoning.lower()
    
    def test_parse_valid_multiple_instructions(self):
        """Test parsing valid JSON with multiple instructions."""
        json_text = '''
        [
          {
            "rank": 1,
            "injection_window": {
              "start_timestamp": 1.5,
              "end_timestamp": 2.0
            },
            "adversarial_instruction": "Turn left immediately",
            "target_module_logic": "Override lane keeping",
            "reasoning": "Primary critical moment"
          },
          {
            "rank": 2,
            "injection_window": {
              "start_timestamp": 3.0,
              "end_timestamp": 3.5
            },
            "adversarial_instruction": "Accelerate now",
            "target_module_logic": "Override speed control",
            "reasoning": "Secondary critical moment"
          },
          {
            "rank": 3,
            "injection_window": {
              "start_timestamp": 5.0,
              "end_timestamp": 5.8
            },
            "adversarial_instruction": "Change lane right",
            "target_module_logic": "Override perception of blind spot",
            "reasoning": "Tertiary critical moment"
          }
        ]
        '''
        
        instruction_set = parse_instructions_from_json(json_text)
        
        assert instruction_set is not None
        assert len(instruction_set.instructions) == 3
        
        assert instruction_set.instructions[0].rank == 1
        assert instruction_set.instructions[1].rank == 2
        assert instruction_set.instructions[2].rank == 3
    
    def test_parse_empty_string(self):
        """Test parsing empty string."""
        instruction_set = parse_instructions_from_json("")
        assert instruction_set is None
    
    def test_parse_empty_array(self):
        """Test parsing empty JSON array."""
        json_text = "[]"
        instruction_set = parse_instructions_from_json(json_text)
        
        assert instruction_set is not None
        assert len(instruction_set.instructions) == 0
    
    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        json_text = "{ not valid json }"
        instruction_set = parse_instructions_from_json(json_text)
        assert instruction_set is None
    
    def test_parse_json_object_instead_of_array(self):
        """Test parsing JSON object when array is expected."""
        json_text = '''
        {
            "rank": 1,
            "injection_window": {
              "start_timestamp": 1.5,
              "end_timestamp": 2.0
            },
            "adversarial_instruction": "Turn left",
            "target_module_logic": "Override planning",
            "reasoning": "Critical moment"
        }
        '''
        instruction_set = parse_instructions_from_json(json_text)
        assert instruction_set is None
    
    def test_parse_missing_required_fields(self):
        """Test parsing with missing required fields."""
        json_text = '''
        [
          {
            "rank": 1,
            "adversarial_instruction": "Turn left"
          }
        ]
        '''
        instruction_set = parse_instructions_from_json(json_text)
        # Should return None since no valid instructions could be parsed
        assert instruction_set is None
    
    def test_parse_partial_valid_instructions(self):
        """Test parsing mix of valid and invalid instructions."""
        json_text = '''
        [
          {
            "rank": 1,
            "injection_window": {
              "start_timestamp": 1.5,
              "end_timestamp": 2.0
            },
            "adversarial_instruction": "Turn left",
            "target_module_logic": "Override planning",
            "reasoning": "Valid instruction"
          },
          {
            "rank": 2,
            "adversarial_instruction": "Invalid - missing fields"
          },
          {
            "rank": 3,
            "injection_window": {
              "start_timestamp": 5.0,
              "end_timestamp": 5.5
            },
            "adversarial_instruction": "Accelerate",
            "target_module_logic": "Override speed",
            "reasoning": "Another valid instruction"
          }
        ]
        '''
        instruction_set = parse_instructions_from_json(json_text)
        
        assert instruction_set is not None
        # Only 2 valid instructions should be parsed
        assert len(instruction_set.instructions) == 2
        assert instruction_set.instructions[0].rank == 1
        assert instruction_set.instructions[1].rank == 3
    
    def test_parse_with_numeric_strings(self):
        """Test parsing with numeric values as strings."""
        json_text = '''
        [
          {
            "rank": "1",
            "injection_window": {
              "start_timestamp": "1.5",
              "end_timestamp": "2.0"
            },
            "adversarial_instruction": "Turn left",
            "target_module_logic": "Override planning",
            "reasoning": "Critical moment"
          }
        ]
        '''
        instruction_set = parse_instructions_from_json(json_text)
        
        assert instruction_set is not None
        assert len(instruction_set.instructions) == 1
        # Should be converted to proper types
        assert instruction_set.instructions[0].rank == 1
        assert instruction_set.instructions[0].injection_window.start_timestamp == 1.5
