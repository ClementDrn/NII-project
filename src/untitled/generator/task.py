import json
from pathlib import Path
from typing import Optional, List, Any
from dataclasses import dataclass
from asyncio import Future
import yaml

from untitled.core.base import CONFIG, PROJECT_ROOT
from untitled.generator.instruction import InstructionResult
from untitled.generator.prompt_dispatcher import PromptRequest


@dataclass
class TaskData:
    """
    Data container passed to and from tasks.
    
    This unified data structure allows tasks to receive and produce
    consistent data types regardless of which specific task is being executed.
    Some fields may be None depending on the task requirements.
    """
    # Input fields
    images: Optional[List[Path]] = None  # List of image paths
    frames: Optional[List[Any]] = None  # List of SimulationFrameData with metadata and timestamps
    descriptions: Optional[str] = None  # Scene descriptions (for multi-step)
    failure_text: Optional[str] = None  # Description of failure that occurred
    was_route_completed: Optional[bool] = None  # Whether route was completed
    
    # Output fields
    instruction: Optional[str] = None  # Generated instruction text
    output_descriptions: Optional[str] = None  # Generated scene descriptions


class Task:
    """
    A Task represents a configurable prompt-based operation in the generation pipeline.
    
    Tasks are defined by YAML configuration files that specify:
    - The path to the system prompt text file
    - What inputs the task requires
    - What outputs the task produces
    
    This allows for a data-oriented generation pipeline where different tasks
    can be composed dynamically based on experiment nature (adversarial vs corrective).
    """
    
    def __init__(self, config_path: str | Path):
        """
        Initialize a Task from a YAML configuration file.
        
        Args:
            config_path: Path to the task YAML configuration file.
                         Can be absolute or relative to PROJECT_ROOT/assets/tasks/
        """
        # Resolve config path
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / "assets" / "tasks" / config_path
        
        if not config_path.exists():
            raise FileNotFoundError(f"Task config file not found: {config_path}")
        
        # Load YAML configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Store config path for debugging
        self._config_path = config_path
        
        # Load task name and description
        self.name: str = config.get('name', config_path.stem)
        self.description: str = config.get('description', '')
        
        # Load system prompt path and content
        prompt_path_str = config.get('system_prompt_path', '')
        if not prompt_path_str:
            raise ValueError(f"Task config '{config_path}' missing required 'system_prompt_path'")
        
        prompt_path = Path(prompt_path_str)
        if not prompt_path.is_absolute():
            prompt_path = PROJECT_ROOT / prompt_path
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        
        with open(prompt_path, 'r') as f:
            self._system_prompt_text = f.read()

        # Prompt formatting
        # NOTE: This is rudimentary right now, can be extended later as needed
        # Replace "${instruction_count_per_generation}$" with actual value from config if present
        # CONFIG.generator is set per-job and is a GeneratorConfig object (or None)
        instruction_count: int = CONFIG.generator.instruction_count_per_generation if CONFIG.generator else 1
        self._system_prompt_text = self._system_prompt_text.replace(
            "${instruction_count_per_generation}$",
            str(instruction_count)
        )
        
        # Load input flags
        inputs = config.get('inputs', {})
        self.has_images_input: bool = inputs.get('images', False)
        self.has_metadata_input: bool = inputs.get('metadata', False)
        self.has_descriptions_input: bool = inputs.get('descriptions', False)
        self.has_failure_text_input: bool = inputs.get('failure_text', False)
        self.has_route_completion_input: bool = inputs.get('route_completion', False)
        self.has_previously_tried_instructions_input: bool = inputs.get('previously_tried_instructions', False)
        
        # Load output flags
        outputs = config.get('outputs', {})
        self.has_instruction_output: bool = outputs.get('instruction', False)
        self.has_descriptions_output: bool = outputs.get('descriptions', False)
    
    def get_prompt_system_text(self) -> str:
        """
        Get the system prompt text for this task.
        
        Returns:
            The system prompt text loaded from the configured file.
        """
        return self._system_prompt_text
    
    def prepare_prompt(
        self,
        future: Future,
        images: Optional[List[Path]] = None,
        frames: Optional[List[Any]] = None,
        descriptions: Optional[str] = None,
        failure_text: Optional[str] = None,
        was_route_completed: Optional[bool] = None,
        previously_tried_instructions: Optional[List[InstructionResult]] = None
    ) -> PromptRequest:
        """
        Prepare a complete PromptRequest based on task inputs.
        
        This method assembles the system text, user text, and image paths
        into a PromptRequest ready to be dispatched.
        
        Args:
            future: The asyncio Future to resolve when the response is ready
            images: List of image paths
            frames: List of SimulationFrameData objects containing metadata and timestamps
            descriptions: Scene descriptions (for multi-step pipelines)
            failure_text: Description of the failure that occurred (for corrective tasks)
            was_route_completed: Whether the route was completed (for corrective tasks)
            previously_tried_instructions: List of previous instruction results (for incremental generation)
        Returns:
            A PromptRequest ready to be dispatched.
        """
        parts = []
        
        # Add failure context for corrective tasks
        if self.has_failure_text_input and failure_text:
            parts.append(f"FAILURE REPORT:\n{failure_text}")
        
        # Add route completion status
        if self.has_route_completion_input and was_route_completed is not None:
            status = "completed" if was_route_completed else "not completed"
            parts.append(f"ROUTE STATUS: The route was {status}.")

        # Add previously tried instructions and their results for incremental generation
        if self.has_previously_tried_instructions_input and previously_tried_instructions:
            previous_instructions_text = "PREVIOUSLY TRIED INSTRUCTIONS:\n"
            for i, instruction_result in enumerate(previously_tried_instructions):
                json_result = {
                    "text": instruction_result.instruction.text,
                    "injection_window": {
                        "start_timestamp": instruction_result.instruction.injection_window.start_timestamp,
                        "end_timestamp": instruction_result.instruction.injection_window.end_timestamp
                    },
                    "reasoning": instruction_result.instruction.reasoning,
                    "was_successful": instruction_result.was_successful
                }
                previous_instructions_text += f"Instruction {i+1}: {json.dumps(json_result)}\n"
            parts.append(previous_instructions_text)

        # Add scene descriptions (for LLM-only planning step)
        if self.has_descriptions_input and descriptions:
            parts.append(descriptions)
        
        # Add metadata (for VLM steps that process images + metadata)
        # Concatenate metadata from frames with frame labels
        if self.has_metadata_input and frames:
            metadata_parts = []
            for i, frame in enumerate(frames):
                metadata_text = str(frame.metadata)
                metadata_parts.append(f"Frame {i+1} (time: {frame.timestamp}):\n{metadata_text}")
            concatenated_metadata = "\n\n".join(metadata_parts)
            parts.append(concatenated_metadata)
        
        user_text = "\n\n".join(parts)
        
        return PromptRequest(
            system_text=self._system_prompt_text,
            user_text=user_text,
            user_image_paths=images if self.has_images_input else None,
            future=future
        )
    
    def get_required_image_paths(self, data: TaskData) -> Optional[List[Path]]:
        """
        Get the image paths required for this task, if any.
        
        Args:
            data: The task data containing potential images
            
        Returns:
            List of image paths if this task requires images, None otherwise.
        """
        if self.has_images_input:
            return data.images
        return None
    
    def __repr__(self) -> str:
        inputs = []
        if self.has_images_input:
            inputs.append('images')
        if self.has_metadata_input:
            inputs.append('metadata')
        if self.has_descriptions_input:
            inputs.append('descriptions')
        if self.has_failure_text_input:
            inputs.append('failure_text')
        if self.has_route_completion_input:
            inputs.append('route_completion')
        if self.has_previously_tried_instructions_input:
            inputs.append('previously_tried_instructions')
        
        outputs = []
        if self.has_instruction_output:
            outputs.append('instruction')
        if self.has_descriptions_output:
            outputs.append('descriptions')
        
        return f"Task(name='{self.name}', inputs={inputs}, outputs={outputs})"
