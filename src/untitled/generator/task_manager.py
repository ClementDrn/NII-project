from dataclasses import dataclass
from typing import Optional, List
from untitled.generator.experiment_nature import ExperimentNature
from untitled.generator.task import Task
from untitled.core.base import LOG


@dataclass
class TaskSet:
    """
    Container for tasks used in a generation pipeline.
    
    For single-step pipelines: only planner_task is used.
    For multi-step pipelines: both descriptor_task and planner_task are used.
    """
    planner_task: Task
    descriptor_task: Optional[Task] = None


class TaskManager:
    """
    Factory class for creating tasks based on pipeline configuration.
    
    This centralizes all task selection logic, making it easier to maintain
    and extend as new task types are added.
    """
    
    # Task filename mappings
    # Single-step planner tasks
    SINGLE_STEP_TASKS = {
        (ExperimentNature.ADVERSARIAL, False): "single_step_adversarial_planner.yaml",
        (ExperimentNature.ADVERSARIAL, True): "incremental_single_step_adversarial_planner.yaml",
        (ExperimentNature.CORRECTIVE, False): "single_step_corrective_planner.yaml",
        # (ExperimentNature.CORRECTIVE, True): "incremental_single_step_corrective_planner.yaml",  # Not yet implemented
    }
    
    # Multi-step planner tasks
    MULTI_STEP_TASKS = {
        (ExperimentNature.ADVERSARIAL, False): "multi_step_adversarial_planner.yaml",
        (ExperimentNature.CORRECTIVE, False): "multi_step_corrective_planner.yaml",
        # Incremental multi-step tasks can be added here when implemented
    }
    
    # Descriptor task (used in multi-step pipelines)
    DESCRIPTOR_TASK = "scene_descriptor.yaml"
    
    @classmethod
    def create_task(
        cls,
        experiment_nature: ExperimentNature,
        is_incremental: bool = False,
        has_descriptor: bool = False
    ) -> TaskSet:
        """
        Create the appropriate task(s) for a generation pipeline.
        
        Args:
            experiment_nature: The experiment nature (ADVERSARIAL, CORRECTIVE)
            is_incremental: Whether to use incremental generation mode
            has_descriptor: Whether the pipeline uses a descriptor step (multi-step)
        
        Returns:
            TaskSet containing the appropriate task(s) for the pipeline
        
        Raises:
            NotImplementedError: If the requested task combination is not yet implemented
            ValueError: If the experiment nature is invalid
        """
        # Select the appropriate task mapping
        task_mapping = cls.MULTI_STEP_TASKS if has_descriptor else cls.SINGLE_STEP_TASKS
        
        # Look up the task filename
        key = (experiment_nature, is_incremental)
        if key not in task_mapping:
            pipeline_type = "multi-step" if has_descriptor else "single-step"
            mode = "incremental" if is_incremental else "standard"
            raise NotImplementedError(
                f"{mode.capitalize()} {pipeline_type} generation is not yet implemented "
                f"for {experiment_nature.value} experiment nature."
            )
        
        task_filename = task_mapping[key]
        planner_task = Task(task_filename)
        LOG.debug(f"Created planner task: {task_filename}")
        
        # Create descriptor task if needed
        descriptor_task = None
        if has_descriptor:
            descriptor_task = Task(cls.DESCRIPTOR_TASK)
            LOG.debug(f"Created descriptor task: {cls.DESCRIPTOR_TASK}")
        
        return TaskSet(
            planner_task=planner_task,
            descriptor_task=descriptor_task
        )
    
    @classmethod
    def create_planner_task(
        cls,
        experiment_nature: ExperimentNature,
        is_incremental: bool = False,
        has_descriptor: bool = False
    ) -> Task:
        """
        Create only the planner task for a generation pipeline.
        
        Convenience method when only the planner task is needed.
        
        Args:
            experiment_nature: The experiment nature (ADVERSARIAL, CORRECTIVE)
            is_incremental: Whether to use incremental generation mode
            has_descriptor: Whether the pipeline uses a descriptor step (multi-step)
        
        Returns:
            The appropriate planner Task
        """
        task_set = cls.create_task(experiment_nature, is_incremental, has_descriptor)
        return task_set.planner_task
    
    @classmethod
    def create_descriptor_task(cls) -> Task:
        """
        Create the descriptor task for multi-step pipelines.
        
        Returns:
            The scene descriptor Task
        """
        task = Task(cls.DESCRIPTOR_TASK)
        LOG.debug(f"Created descriptor task: {cls.DESCRIPTOR_TASK}")
        return task
