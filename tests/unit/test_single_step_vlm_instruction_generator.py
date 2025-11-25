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
    mock_load_config.return_value = mock_config
    
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    from untitled.simulator.simulation_data import SimulationData, SimulationFrameData, SimulationMetadata


# --- Fixtures and Mocks ------------------------------------------------------

@pytest.fixture
def mock_simulation_data_single_frame():
    """Load mock simulation data from data.json with a single frame."""
    # Load the JSON file
    data_json_path = Path(__file__).parent.parent / "data" / "nuScenes-0" / "data.json"
    with open(data_json_path, "r") as f:
        data_dict = json.load(f)
    
    # Resolve the absolute path for image_dir
    image_dir = (data_json_path.parent / data_dict["image_dir"]).resolve()
    
    # Create SimulationData object with only the first frame
    frames = [
        SimulationFrameData(
            image_name=data_dict["frames"][0]["image_name"],
            metadata=SimulationMetadata(**data_dict["frames"][0]["metadata"])
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

@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task message")  # mock_file
def test_init(mock_file, mock_parse, mock_vlm_class):
    """Test initialization of SingleStepVLMInstructionGenerator."""
    generator = SingleStepVLMInstructionGenerator()
    
    # Verify VLM was instantiated
    mock_vlm_class.assert_called_once()
    
    # Verify system task message was loaded
    assert generator.system_task_message == "Test system task message"
    mock_file.assert_called_once()

@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_generate_instruction_single_frame(mock_file, mock_parse, mock_vlm_class, mock_simulation_data_single_frame, mock_vlm_response):
    """Test generating instruction from a single frame."""
    # Setup mock VLM
    mock_vlm_instance = Mock()
    mock_vlm_instance.prompt.return_value = mock_vlm_response
    mock_vlm_class.return_value = mock_vlm_instance
    
    # Setup parse_vlm_response to return parsed instruction
    mock_parse.return_value = "Turn left at the intersection."
    
    # Create generator and generate instruction
    generator = SingleStepVLMInstructionGenerator()
    instruction = generator.generate_instruction_from_images(mock_simulation_data_single_frame)
    
    # Verify the instruction was parsed correctly (without the <think> tag)
    assert instruction == "Turn left at the intersection."
    
    # Verify VLM prompt was called with correct arguments
    mock_vlm_instance.prompt.assert_called_once()
    call_args = mock_vlm_instance.prompt.call_args
    assert call_args[0][0] == "Test system task"  # system_task_message
    assert "speed" in call_args[0][1].lower()  # metadata_text contains speed
    assert len(call_args[0][2]) == 1  # one image path

@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_generate_instruction_multiple_frames_raises_error(mock_file, mock_parse, mock_vlm_class):
    """Test that generating instruction from multiple frames raises NotImplementedError."""
    # Create multi-frame simulation data
    frames = [
        SimulationFrameData(
            image_name="image1.jpg",
            metadata=SimulationMetadata(speed=0)
        ),
        SimulationFrameData(
            image_name="image2.jpg",
            metadata=SimulationMetadata(speed=0)
        )
    ]
    multi_frame_data = SimulationData(
        image_dir=Path("/tmp/images"),
        frame_count=2,
        frames=frames
    )
    
    mock_vlm_class.return_value = Mock()
    generator = SingleStepVLMInstructionGenerator()
    
    # Verify that NotImplementedError is raised
    with pytest.raises(NotImplementedError, match="Currently only single image and metadata is supported"):
        generator.generate_instruction_from_images(multi_frame_data)

@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_generate_instruction_no_think_tags(mock_file, mock_parse, mock_vlm_class, mock_simulation_data_single_frame):
    """Test parsing instruction when response has no <think> tags."""
    # Setup mock VLM with response without think tags
    mock_vlm_instance = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Continue straight ahead."
    mock_vlm_instance.prompt.return_value = mock_response
    mock_vlm_class.return_value = mock_vlm_instance
    
    # Setup parse_vlm_response to return the content as-is (simulating no think tags)
    mock_parse.return_value = "Continue straight ahead."
    
    # Create generator and generate instruction
    generator = SingleStepVLMInstructionGenerator()
    instruction = generator.generate_instruction_from_images(mock_simulation_data_single_frame)
    
    # Verify the instruction is returned as-is
    assert instruction == "Continue straight ahead."

@patch("untitled.generator.single_step_vlm_instruction_generator.VLM")  # mock_vlm_class
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")  # mock_parse
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")  # mock_file
def test_free_method(mock_file, mock_parse, mock_vlm_class):
    """Test the free method (should do nothing)."""
    mock_vlm_class.return_value = Mock()
    generator = SingleStepVLMInstructionGenerator()
    
    # Should not raise any errors
    generator.free()
    assert True
