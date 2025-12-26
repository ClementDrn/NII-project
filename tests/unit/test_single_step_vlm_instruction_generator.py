import pytest
from unittest.mock import Mock, patch, mock_open, MagicMock, AsyncMock
from pathlib import Path
import json
import asyncio

from untitled.simulator.simulation_data import SimulationData, SimulationFrameData, SimulationMetadata
from untitled.generator.vlm import VLMClient
from untitled.core.settings import GeneratorStepConfig, ModelInstanceConfig, LocalVLMConfig


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

@patch("untitled.generator.single_step_vlm_instruction_generator.PromptDispatcher")
@patch("untitled.generator.single_step_vlm_instruction_generator.VLMClientFactory")
@patch("untitled.generator.vlm.get_model_config")
@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")
@patch("builtins.open", new_callable=mock_open, read_data="Test system task message")
def test_init(mock_file, mock_config, mock_get_model_config, mock_factory, mock_dispatcher):
    """Test initialization of SingleStepVLMInstructionGenerator."""
    # Setup CONFIG with proper steps and model_instances
    step = GeneratorStepConfig(
        id="step-1",
        model_name="test-model",
        temperature=0.7,
        max_tokens=512
    )
    mock_config.generator.steps = [step]
    
    instance = ModelInstanceConfig(
        id="instance-1",
        model_name="test-model",
        type="vlm",
        step_ids=["step-1"],
        local_config=LocalVLMConfig(port=8000)
    )
    mock_config.model_instances = [instance]
    
    mock_model_config = MagicMock()
    mock_get_model_config.return_value = mock_model_config
    
    mock_vlm = MagicMock()
    mock_vlm.is_parallelizable.return_value = False
    mock_factory.create.return_value = mock_vlm
    
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    generator = SingleStepVLMInstructionGenerator()
    
    # Verify VLM client was created
    mock_factory.create.assert_called_once()
    
    # Verify system task message was loaded
    assert generator._system_task_message == "Test system task message"
    mock_file.assert_called_once()

@pytest.mark.asyncio
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_instructions_from_json")
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")
@patch("untitled.generator.single_step_vlm_instruction_generator.PromptDispatcher")
@patch("untitled.generator.single_step_vlm_instruction_generator.VLMClientFactory")
@patch("untitled.generator.vlm.get_model_config")
@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")
async def test_generate_instruction_single_frame(mock_file, mock_config, mock_get_model_config, mock_factory, 
                                                  mock_dispatcher, mock_parse, mock_parse_instruction, 
                                                  mock_simulation_data_single_frame):
    """Test generating instruction from a single frame."""
    from untitled.generator.instruction import Instruction, InstructionSet, InjectionWindow
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    
    # Setup CONFIG with proper steps and model_instances
    step = GeneratorStepConfig(
        id="step-1",
        model_name="test-model",
        temperature=0.7,
        max_tokens=512
    )
    mock_config.generator.steps = [step]
    
    instance = ModelInstanceConfig(
        id="instance-1",
        model_name="test-model",
        type="vlm",
        step_ids=["step-1"],
        local_config=LocalVLMConfig(port=8000)
    )
    mock_config.model_instances = [instance]
    
    mock_get_model_config.return_value = MagicMock()
    
    # Mock VLM client
    mock_vlm = MagicMock()
    mock_vlm.is_parallelizable.return_value = False
    mock_vlm.get_status.return_value = MagicMock()
    mock_factory.create.return_value = mock_vlm

    # Mock prompt dispatcher
    json_response = '''[{
        "rank": 1,
        "injection_window": {"start_timestamp": 1.5, "end_timestamp": 2.0},
        "adversarial_instruction": "Turn left at the intersection",
        "target_module_logic": "Override lane keeping",
        "reasoning": "Critical moment"
    }]'''

    mock_dispatcher_instance = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_instance

    # Setup parse_vlm_response
    mock_parse.return_value = json_response

    # Setup parse_instructions_from_json
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

    # Mock the dispatcher's add method to immediately resolve the future
    async def mock_add_side_effect(prompt_request):
        prompt_request.future.set_result(f"<think>Analyzing...</think>\n\n{json_response}")
    
    generator._prompt_dispatcher.add = mock_add_side_effect

    instruction_set = await generator.generate_instructions_from_images(mock_simulation_data_single_frame)

    # Verify the instruction set was returned
    assert instruction_set == expected_instruction_set
    assert len(instruction_set.instructions) == 1
    assert instruction_set.instructions[0].adversarial_instruction == "Turn left at the intersection"

@pytest.mark.asyncio
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_instructions_from_json")
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")
@patch("untitled.generator.single_step_vlm_instruction_generator.PromptDispatcher")
@patch("untitled.generator.single_step_vlm_instruction_generator.VLMClientFactory")
@patch("untitled.generator.vlm.get_model_config")
@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")
async def test_generate_instruction_multiple_frames_uses_first_frame(mock_file, mock_config, mock_get_model_config,
                                                                      mock_factory, mock_dispatcher, mock_parse,
                                                                      mock_parse_instruction):
    """Test that generating instruction with insufficient frames falls back to first frame."""
    from untitled.generator.instruction import Instruction, InstructionSet, InjectionWindow
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    
    # Setup CONFIG with proper steps and model_instances
    step = GeneratorStepConfig(
        id="step-1",
        model_name="test-model",
        temperature=0.7,
        max_tokens=512
    )
    mock_config.generator.steps = [step]
    
    instance = ModelInstanceConfig(
        id="instance-1",
        model_name="test-model",
        type="vlm",
        step_ids=["step-1"],
        local_config=LocalVLMConfig(port=8000)
    )
    mock_config.model_instances = [instance]
    
    mock_get_model_config.return_value = MagicMock()
    
    # Mock VLM client
    mock_vlm = MagicMock()
    mock_vlm.is_parallelizable.return_value = False
    mock_vlm.get_status.return_value = MagicMock()
    mock_factory.create.return_value = mock_vlm
    
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
    json_response = '''[{
        "rank": 1,
        "injection_window": {"start_timestamp": 0.0, "end_timestamp": 0.1},
        "adversarial_instruction": "Test instruction",
        "target_module_logic": "Test logic",
        "reasoning": "Test reasoning"
    }]'''

    mock_dispatcher_instance = MagicMock()
    mock_dispatcher.return_value = mock_dispatcher_instance

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

    # Mock the dispatcher's add method to immediately resolve the future
    async def mock_add_side_effect(prompt_request):
        prompt_request.future.set_result(json_response)
    
    generator._prompt_dispatcher.add = mock_add_side_effect

    # Should use first frame without raising an error
    instruction_set = await generator.generate_instructions_from_images(multi_frame_data)

    # Verify it returns a valid instruction set
    assert instruction_set is not None
    assert len(instruction_set.instructions) == 1

@pytest.mark.asyncio
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_instructions_from_json")
@patch("untitled.generator.single_step_vlm_instruction_generator.parse_vlm_response")
@patch("untitled.generator.single_step_vlm_instruction_generator.PromptDispatcher")
@patch("untitled.generator.single_step_vlm_instruction_generator.VLMClientFactory")
@patch("untitled.generator.vlm.get_model_config")
@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")
async def test_generate_instruction_no_think_tags(mock_file, mock_config, mock_get_model_config, mock_factory,
                                                   mock_dispatcher, mock_parse, mock_parse_instruction,
                                                   mock_simulation_data_single_frame):
    """Test parsing instruction when response has no <think> tags."""
    from untitled.generator.instruction import Instruction, InstructionSet, InjectionWindow
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    
    # Setup CONFIG with proper steps and model_instances
    step = GeneratorStepConfig(
        id="step-1",
        model_name="test-model",
        temperature=0.7,
        max_tokens=512
    )
    mock_config.generator.steps = [step]
    
    instance = ModelInstanceConfig(
        id="instance-1",
        model_name="test-model",
        type="vlm",
        step_ids=["step-1"],
        local_config=LocalVLMConfig(port=8000)
    )
    mock_config.model_instances = [instance]
    
    mock_get_model_config.return_value = MagicMock()
    
    # Mock VLM client
    mock_vlm = MagicMock()
    mock_vlm.is_parallelizable.return_value = False
    mock_vlm.get_status.return_value = MagicMock()
    mock_factory.create.return_value = mock_vlm
    
    # Setup mock VLM with response without think tags
    json_response = '''[{
        "rank": 1,
        "injection_window": {"start_timestamp": 2.0, "end_timestamp": 2.5},
        "adversarial_instruction": "Continue straight ahead",
        "target_module_logic": "Override path planning",
        "reasoning": "Testing moment"
    }]'''
    
    mock_dispatcher_instance = MagicMock()
    mock_dispatcher_instance.add = AsyncMock()
    mock_dispatcher_instance.flush = AsyncMock()
    mock_dispatcher.return_value = mock_dispatcher_instance
    
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
    
    # Mock the prompt request to return our future
    async def set_future(prompt_request):
        prompt_request.future.set_result(json_response)

    with patch.object(generator._prompt_dispatcher, 'add', new_callable=AsyncMock) as mock_add:
        mock_add.side_effect = set_future
        instruction_set = await generator.generate_instructions_from_images(mock_simulation_data_single_frame)
    
    # Verify the instruction set was returned
    assert instruction_set == expected_instruction_set
    assert instruction_set.instructions[0].adversarial_instruction == "Continue straight ahead"

@pytest.mark.asyncio
@patch("untitled.generator.single_step_vlm_instruction_generator.PromptDispatcher")
@patch("untitled.generator.single_step_vlm_instruction_generator.VLMClientFactory")
@patch("untitled.generator.vlm.get_model_config")
@patch("untitled.generator.single_step_vlm_instruction_generator.CONFIG")
@patch("builtins.open", new_callable=mock_open, read_data="Test system task")
async def test_free_method(mock_file, mock_config, mock_get_model_config, mock_factory, mock_dispatcher):
    """Test the free method shuts down dispatcher and stops VLM."""
    from untitled.generator.single_step_vlm_instruction_generator import SingleStepVLMInstructionGenerator
    
    # Setup CONFIG with proper steps and model_instances
    step = GeneratorStepConfig(
        id="step-1",
        model_name="test-model",
        temperature=0.7,
        max_tokens=512
    )
    mock_config.generator.steps = [step]
    
    instance = ModelInstanceConfig(
        id="instance-1",
        model_name="test-model",
        type="vlm",
        step_ids=["step-1"],
        local_config=LocalVLMConfig(port=8000)
    )
    mock_config.model_instances = [instance]
    
    mock_get_model_config.return_value = MagicMock()
    
    # Mock VLM client
    mock_vlm = MagicMock()
    mock_vlm.is_parallelizable.return_value = False
    mock_vlm.get_state.return_value = VLMClient.State.STARTED
    mock_vlm.close_connection = AsyncMock()
    mock_vlm.stop = AsyncMock()
    mock_factory.create.return_value = mock_vlm

    # Mock dispatcher
    mock_dispatcher_instance = MagicMock()
    mock_dispatcher_instance.stop = AsyncMock()
    mock_dispatcher.return_value = mock_dispatcher_instance

    generator = SingleStepVLMInstructionGenerator()

    # Test free method
    await generator.free()

    # Verify dispatcher stop and VLM stop were called
    mock_dispatcher_instance.stop.assert_called_once()
    mock_vlm.stop.assert_called_once()