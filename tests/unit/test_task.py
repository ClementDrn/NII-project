import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import yaml
from untitled.generator.task import Task, TaskData


@dataclass
class MockMetadata:
    """Mock metadata for testing."""
    speed: float = 10.0
    position: list = None
    
    def __str__(self):
        return f"speed={self.speed}, position={self.position or [0, 0, 0]}"


@dataclass  
class MockFrame:
    """Mock frame data for testing."""
    image_name: str = "frame.png"
    metadata: Any = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = MockMetadata()


@pytest.fixture
def temp_dir():
    """Create a temporary directory and clean up after test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_task_files(temp_dir):
    """Create sample task configuration and prompt files."""
    # Create prompt file
    prompts_dir = temp_dir / "assets" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / "test_prompt.txt"
    prompt_file.write_text("You are a test assistant.\n\nFollow these instructions carefully.")
    
    # Create task config file
    tasks_dir = temp_dir / "assets" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / "test_task.yaml"
    task_data = {
        "name": "Test Task",
        "description": "A test task for unit testing",
        "system_prompt_path": str(prompt_file),
        "inputs": {
            "images": True,
            "metadata": True,
            "descriptions": False,
            "failure_text": False,
            "route_completion": False
        },
        "outputs": {
            "instruction": True,
            "descriptions": False
        }
    }
    with open(task_file, 'w') as f:
        yaml.dump(task_data, f)
    
    return task_file, prompt_file


@pytest.fixture
def corrective_task_files(temp_dir):
    """Create sample corrective task configuration files."""
    # Create prompt file
    prompts_dir = temp_dir / "assets" / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = prompts_dir / "corrective_prompt.txt"
    prompt_file.write_text("You are a safety intervention specialist.")
    
    # Create task config file
    tasks_dir = temp_dir / "assets" / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    task_file = tasks_dir / "corrective_task.yaml"
    task_data = {
        "name": "Corrective Task",
        "description": "A corrective task for testing",
        "system_prompt_path": str(prompt_file),
        "inputs": {
            "images": True,
            "metadata": True,
            "descriptions": False,
            "failure_text": True,
            "route_completion": True
        },
        "outputs": {
            "instruction": True,
            "descriptions": False
        }
    }
    with open(task_file, 'w') as f:
        yaml.dump(task_data, f)
    
    return task_file, prompt_file


class TestTask:
    """Test suite for Task class."""
    
    def test_task_creation(self, sample_task_files):
        """Test creating a Task from a YAML file."""
        task_file, prompt_file = sample_task_files
        task = Task(task_file)
        
        assert task.name == "Test Task"
        assert task.description == "A test task for unit testing"
        assert task.has_images_input is True
        assert task.has_metadata_input is True
        assert task.has_descriptions_input is False
        assert task.has_failure_text_input is False
        assert task.has_route_completion_input is False
        assert task.has_instruction_output is True
        assert task.has_descriptions_output is False
    
    def test_get_prompt_system_text(self, sample_task_files):
        """Test getting system prompt text."""
        task_file, prompt_file = sample_task_files
        task = Task(task_file)
        
        system_text = task.get_prompt_system_text()
        assert "You are a test assistant." in system_text
        assert "Follow these instructions carefully" in system_text
    
    def test_prepare_prompt_with_frames(self, sample_task_files):
        """Test preparing prompt request with frame data."""
        task_file, prompt_file = sample_task_files
        task = Task(task_file)
        
        frames = [
            MockFrame(image_name="frame1.png", metadata=MockMetadata(speed=10.0), timestamp=1.0),
            MockFrame(image_name="frame2.png", metadata=MockMetadata(speed=15.0), timestamp=2.0),
        ]
        
        future = asyncio.Future()
        prompt_request = task.prepare_prompt(
            future=future,
            frames=frames,
            images=[Path("/path/to/image1.png"), Path("/path/to/image2.png")]
        )
        
        # Check the PromptRequest is properly created
        assert prompt_request.future is future
        assert prompt_request.system_text == task.get_prompt_system_text()
        assert "Frame 1 (time: 1.0)" in prompt_request.user_text
        assert "Frame 2 (time: 2.0)" in prompt_request.user_text
        assert "speed=10.0" in prompt_request.user_text
        assert "speed=15.0" in prompt_request.user_text
        assert prompt_request.user_image_paths == [Path("/path/to/image1.png"), Path("/path/to/image2.png")]
    
    def test_prepare_prompt_corrective(self, corrective_task_files):
        """Test preparing prompt request for corrective task with failure info."""
        task_file, prompt_file = corrective_task_files
        task = Task(task_file)
        
        frames = [MockFrame(image_name="frame.png", timestamp=5.0)]
        
        future = asyncio.Future()
        prompt_request = task.prepare_prompt(
            future=future,
            frames=frames,
            failure_text="Collision with vehicle at t=10.5",
            was_route_completed=False,
            images=[Path("/path/to/image.png")]
        )
        
        assert "FAILURE REPORT" in prompt_request.user_text
        assert "Collision with vehicle at t=10.5" in prompt_request.user_text
        assert "not completed" in prompt_request.user_text
        assert prompt_request.user_image_paths == [Path("/path/to/image.png")]
    
    def test_prepare_prompt_route_completed(self, corrective_task_files):
        """Test preparing prompt request with route completed."""
        task_file, prompt_file = corrective_task_files
        task = Task(task_file)
        
        frames = [MockFrame(image_name="frame.png", timestamp=3.0)]
        
        future = asyncio.Future()
        prompt_request = task.prepare_prompt(
            future=future,
            frames=frames,
            failure_text="Red light violation",
            was_route_completed=True,
            images=[Path("/path/to/image.png")]
        )
        
        assert "was completed" in prompt_request.user_text
    
    def test_prepare_prompt_no_images(self, temp_dir):
        """Test preparing prompt request when task has no image input."""
        # Create prompt file
        prompts_dir = temp_dir / "assets" / "prompts"
        prompts_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = prompts_dir / "no_images_prompt.txt"
        prompt_file.write_text("Analyze the descriptions.")
        
        # Create task config file without images input
        tasks_dir = temp_dir / "assets" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_file = tasks_dir / "no_images_task.yaml"
        task_data = {
            "name": "No Images Task",
            "description": "A task without image input",
            "system_prompt_path": str(prompt_file),
            "inputs": {
                "images": False,
                "metadata": False,
                "descriptions": True,
                "failure_text": False,
                "route_completion": False
            },
            "outputs": {
                "instruction": True,
                "descriptions": False
            }
        }
        with open(task_file, 'w') as f:
            yaml.dump(task_data, f)
        
        task = Task(task_file)
        
        future = asyncio.Future()
        prompt_request = task.prepare_prompt(
            future=future,
            descriptions="Some scene descriptions"
        )
        
        # When task has no images input, user_image_paths should be None
        assert prompt_request.user_image_paths is None
        assert "Some scene descriptions" in prompt_request.user_text
    
    def test_task_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for missing config file."""
        with pytest.raises(FileNotFoundError):
            Task(temp_dir / "nonexistent.yaml")
    
    def test_task_prompt_file_not_found(self, temp_dir):
        """Test that FileNotFoundError is raised for missing prompt file."""
        # Create task config pointing to non-existent prompt
        tasks_dir = temp_dir / "assets" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_file = tasks_dir / "bad_task.yaml"
        task_data = {
            "name": "Bad Task",
            "system_prompt_path": "/nonexistent/prompt.txt",
            "inputs": {},
            "outputs": {}
        }
        with open(task_file, 'w') as f:
            yaml.dump(task_data, f)
        
        with pytest.raises(FileNotFoundError):
            Task(task_file)
    
    def test_task_missing_system_prompt_path(self, temp_dir):
        """Test that ValueError is raised for missing system_prompt_path."""
        tasks_dir = temp_dir / "assets" / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        task_file = tasks_dir / "missing_prompt.yaml"
        task_data = {
            "name": "Missing Prompt Task",
            "inputs": {},
            "outputs": {}
        }
        with open(task_file, 'w') as f:
            yaml.dump(task_data, f)
        
        with pytest.raises(ValueError) as exc_info:
            Task(task_file)
        
        assert "system_prompt_path" in str(exc_info.value)
    
    def test_task_repr(self, sample_task_files):
        """Test Task string representation."""
        task_file, prompt_file = sample_task_files
        task = Task(task_file)
        
        repr_str = repr(task)
        assert "Test Task" in repr_str
        assert "images" in repr_str
        assert "metadata" in repr_str
        assert "instruction" in repr_str


class TestTaskData:
    """Test suite for TaskData dataclass."""
    
    def test_task_data_defaults(self):
        """Test TaskData with default values."""
        data = TaskData()
        
        assert data.images is None
        assert data.frames is None
        assert data.descriptions is None
        assert data.failure_text is None
        assert data.was_route_completed is None
        assert data.instruction is None
        assert data.output_descriptions is None
    
    def test_task_data_with_values(self):
        """Test TaskData with values."""
        frames = [MockFrame(image_name="frame.png")]
        data = TaskData(
            images=[Path("/path/to/img.png")],
            frames=frames,
            failure_text="Collision occurred",
            was_route_completed=True
        )
        
        assert len(data.images) == 1
        assert len(data.frames) == 1
        assert data.failure_text == "Collision occurred"
        assert data.was_route_completed is True
