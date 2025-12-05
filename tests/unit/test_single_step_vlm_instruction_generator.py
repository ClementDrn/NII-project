import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import json
import sys

# Mock the CONFIG before importing any modules that depend on it
with patch('untitled.core.settings.load_config') as mock_load_config:
    mock_config = MagicMock()
    mock_config.vlm.host = "http://localhost:8000"
    mock_config.vlm.api_key = "test-key"
    mock_config.generator.models = [MagicMock()]  # One model for single-step
    mock_load_config.return_value = mock_config
    
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    from untitled.simulator.simulation_data import SimulationData, SimulationFrameData, SimulationMetadata


# --- Fixtures and Mocks ------------------------------------------------------

@pytest.fixture
def mock_simulation_data_single_frame():
    """Load mock simulation data from data.json with a single frame."""
    # Load the JSON file
    data_json_path = Path(__file__).parent.parent.parent / "data" / "nuScenes-0" / "data.json"
    with open(data_json_path, "r") as f:
        data_dict = json.load(f)
    
    # Resolve the absolute path for image_dir
    image_dir = (data_json_path.parent / data_dict["image_dir"]).resolve()
    
    # Create SimulationData object with only the first frame
    frames = [
        SimulationFrameData(
            image_name=data_dict["frames"][0]["image_name"],
            metadata=SimulationMetadata(
                step=0,
                control={},
                speed=data_dict["frames"][0]["metadata"]["speed"],
                position=[0, 0, 0],
                rotation=[0, 0, 0],
                forward_vector=[1, 0, 0],
                right_vector=[0, 1, 0],
                velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                acceleration=[0, 0, 0]
            )
        )
    ]
    
    return SimulationData(
        image_dir=image_dir,
        frame_count=1,
        frames=frames
    )


@pytest.fixture
def mock_vlm_response():
    """Mock VLM response object."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "<think>Analyzing the scene...</think>\n\nTurn left at the intersection."
    return mock_response


@pytest.fixture
def mock_system_task():
    """Mock system task message."""
    return "You are a helpful assistant that generates driving instructions."


# --- Tests -------------------------------------------------------------------

@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")  # mock_config
@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task message")  # mock_file
def test_init(mock_file, mock_parse, mock_vlm_class, mock_config):
    """Test initialization of SingleStepVLMInstructionGenerator."""
    mock_config.generator.models = [MagicMock()]
    generator = SingleStepVLMInstructionGenerator()
    
    # Verify VLM was instantiated
    mock_vlm_class.assert_called_once()
    
    # Verify system task message was loaded
    assert generator.system_task_message == "Test system task message"
    mock_file.assert_called_once()

@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")  # mock_config
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_instructions_from_json")  # mock_parse_instruction
@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_generate_instruction_single_frame(mock_file, mock_parse, mock_vlm_class, mock_parse_instruction, mock_config, mock_simulation_data_single_frame, mock_vlm_response):
    """Test generating instruction from a single frame."""
    from untitled.generator.instruction import Instruction, InstructionSet, InjectionWindow
    
    mock_config.generator.models = [MagicMock()]
    # Setup mock VLM
    mock_vlm_instance = Mock()
    json_response = '''[{
        "rank": 1,
        "injection_window": {"start_timestamp": 1.5, "end_timestamp": 2.0},
        "adversarial_instruction": "Turn left at the intersection",
        "target_module_logic": "Override lane keeping",
        "reasoning": "Critical moment"
    }]'''
    mock_vlm_instance.prompt.return_value = f"<think>Analyzing...</think>\n\n{json_response}"
    mock_vlm_class.return_value = mock_vlm_instance
    
    # Setup parse_vlm_response to return JSON response
    mock_parse.return_value = json_response
    
    # Setup parse_instructions_from_json to return an InstructionSet object
    window = InjectionWindow(start_timestamp=1.5, end_timestamp=2.0)
    instruction = Instruction(
        rank=1,
        injection_window=window,
        adversarial_instruction="Turn left at the intersection",
        target_module_logic="Override lane keeping",
        reasoning="Critical moment"
    )
    expected_instruction_set = InstructionSet(instructions=[instruction])
    mock_parse_instruction.return_value = expected_instruction_set
    
    # Create generator and generate instruction
    generator = SingleStepVLMInstructionGenerator()
    instruction_set = generator.generate_instructions_from_images(mock_simulation_data_single_frame)
    
    # Verify the instruction set was returned
    assert instruction_set == expected_instruction_set
    assert len(instruction_set.instructions) == 1
    assert instruction_set.instructions[0].adversarial_instruction == "Turn left at the intersection"
    
    # Verify VLM prompt was called with correct arguments
    mock_vlm_instance.prompt.assert_called_once()
    call_args = mock_vlm_instance.prompt.call_args
    assert call_args[0][0] == "Test system task"  # system_task_message
    assert "speed" in call_args[0][1].lower()  # metadata_text contains speed
    assert len(call_args[0][2]) == 1  # one image path

@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")  # mock_config
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_instructions_from_json")  # mock_parse_instruction
@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_generate_instruction_multiple_frames_uses_first_frame(mock_file, mock_parse, mock_vlm_class, mock_parse_instruction, mock_config):
    """Test that generating instruction with insufficient frames falls back to first frame."""
    from untitled.generator.instruction import Instruction, InstructionSet, InjectionWindow
    
    mock_config.generator.models = [MagicMock()]
    # Create multi-frame simulation data
    frames = [
        SimulationFrameData(
            image_name="image1.jpg",
            metadata=SimulationMetadata(
                step=0,
                control={},
                speed=0,
                position=[0, 0, 0],
                rotation=[0, 0, 0],
                forward_vector=[1, 0, 0],
                right_vector=[0, 1, 0],
                velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                acceleration=[0, 0, 0]
            )
        ),
        SimulationFrameData(
            image_name="image2.jpg",
            metadata=SimulationMetadata(
                step=1,
                control={},
                speed=0,
                position=[0, 0, 0],
                rotation=[0, 0, 0],
                forward_vector=[1, 0, 0],
                right_vector=[0, 1, 0],
                velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                acceleration=[0, 0, 0]
            )
        )
    ]
    multi_frame_data = SimulationData(
        image_dir=Path("/tmp/images"),
        frame_count=2,
        frames=frames
    )
    
    # Setup mocks
    mock_vlm_instance = Mock()
    json_response = '''[{
        "rank": 1,
        "injection_window": {"start_timestamp": 0.0, "end_timestamp": 0.1},
        "adversarial_instruction": "Test instruction",
        "target_module_logic": "Test logic",
        "reasoning": "Test reasoning"
    }]'''
    mock_vlm_instance.prompt.return_value = json_response
    mock_vlm_class.return_value = mock_vlm_instance
    mock_parse.return_value = json_response
    
    # Setup parse_instructions_from_json to return an InstructionSet
    window = InjectionWindow(start_timestamp=0.0, end_timestamp=0.1)
    instruction = Instruction(
        rank=1,
        injection_window=window,
        adversarial_instruction="Test instruction",
        target_module_logic="Test logic",
        reasoning="Test reasoning"
    )
    expected_instruction_set = InstructionSet(instructions=[instruction])
    mock_parse_instruction.return_value = expected_instruction_set
    
    generator = SingleStepVLMInstructionGenerator()
    
    # Should use first frame without raising an error
    instruction_set = generator.generate_instructions_from_images(multi_frame_data)
    
    # Verify it returns a valid instruction set
    assert instruction_set is not None
    assert len(instruction_set.instructions) == 1

@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")  # mock_config
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_instructions_from_json")  # mock_parse_instruction
@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_generate_instruction_no_think_tags(mock_file, mock_parse, mock_vlm_class, mock_parse_instruction, mock_config, mock_simulation_data_single_frame):
    """Test parsing instruction when response has no <think> tags."""
    from untitled.generator.instruction import Instruction, InstructionSet, InjectionWindow
    
    mock_config.generator.models = [MagicMock()]
    # Setup mock VLM with response without think tags
    mock_vlm_instance = Mock()
    json_response = '''[{
        "rank": 1,
        "injection_window": {"start_timestamp": 2.0, "end_timestamp": 2.5},
        "adversarial_instruction": "Continue straight ahead",
        "target_module_logic": "Override path planning",
        "reasoning": "Testing moment"
    }]'''
    mock_vlm_instance.prompt.return_value = json_response
    mock_vlm_class.return_value = mock_vlm_instance
    
    # Setup parse_vlm_response to return JSON (no think tags)
    mock_parse.return_value = json_response
    
    # Setup parse_instructions_from_json to return an InstructionSet object
    window = InjectionWindow(start_timestamp=2.0, end_timestamp=2.5)
    instruction = Instruction(
        rank=1,
        injection_window=window,
        adversarial_instruction="Continue straight ahead",
        target_module_logic="Override path planning",
        reasoning="Testing moment"
    )
    expected_instruction_set = InstructionSet(instructions=[instruction])
    mock_parse_instruction.return_value = expected_instruction_set
    
    # Create generator and generate instruction
    generator = SingleStepVLMInstructionGenerator()
    instruction_set = generator.generate_instructions_from_images(mock_simulation_data_single_frame)
    
    # Verify the instruction set was returned
    assert instruction_set == expected_instruction_set
    assert instruction_set.instructions[0].adversarial_instruction == "Continue straight ahead"

@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")  # mock_config
@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_free_method(mock_file, mock_parse, mock_vlm_class, mock_config):
    """Test the free method (should do nothing)."""
    mock_config.generator.models = [MagicMock()]
    mock_vlm_class.return_value = Mock()
    generator = SingleStepVLMInstructionGenerator()
    
    # Should not raise any errors
    generator.free()
    assert True
