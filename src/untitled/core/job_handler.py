from untitled.core.base import LOG
from untitled.core.version import JOBS_VERSION, are_versions_compatible
from untitled.generator.experiment_nature import ExperimentNature

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml


@dataclass
class Job:
    """
    Represents a job containing scenarios to run with specific settings.
    
    Attributes:
        id: Unique identifier for the job (auto-assigned)
        name: Human-readable name for the job
        simulator: ID of the simulator config to use (from config.yaml)
        scenarios: List of scenario IDs to run
        ground_truth_count: Number of baseline runs without instructions
        ground_truth_cache_dir: Optional path to directory containing cached ground truth logs
        filter: ID of the filter config to use (from config.yaml)
        generator: ID of the generator config to use (from config.yaml)
        model_instances: List of model instance IDs to use (from config.yaml)
        experiment_nature: The nature of experiments (adversarial or corrective)
        instruction_generation_count: Number of instruction generation calls to make
        instruction_count_per_generation: Number of instructions to use per generation call
        results: Dictionary storing simulation results keyed by (scenario_id, run_id)
    """
    id: int
    name: str
    simulator: str
    scenarios: List[int]
    ground_truth_count: int = 1
    ground_truth_cache_dir: str = ""
    filter: str = ""
    generator: str = ""
    model_instances: List[str] = field(default_factory=list)
    experiment_nature: ExperimentNature = ExperimentNature.ADVERSARIAL
    instruction_generation_count: int = 1
    instruction_count_per_generation: int = 1
    results: Dict[tuple, Any] = field(default_factory=dict)


class JobHandler:
    """
    Handles loading and managing jobs from a YAML configuration file.
    
    Jobs define which scenarios to run and how many times to run them,
    with both ground truth (baseline) and experiment (VLM-instructed) runs.
    """
    
    def __init__(self, file_path: str = ""):
        """
        Initialize the JobHandler and load jobs from file.
        
        Args:
            file_path: Path to the jobs YAML file
        """
        self.jobs: List[Job] = []
        if file_path:
            self._load_jobs_from_file(file_path)
        else:
            LOG.warning("No job file path provided. No jobs loaded.")

    def _load_jobs_from_file(self, file_path: str):
        """
        Load jobs from a YAML file.

        Args:
            file_path: Path to the YAML file
            
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the YAML structure is invalid
        """
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            LOG.error(f"Job file not found: {file_path}")
            raise FileNotFoundError(f"Job file not found: {file_path}")
        
        # Load YAML file
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            LOG.error(f"Failed to parse YAML file: {e}")
            raise ValueError(f"Invalid YAML format in {file_path}: {e}")
        
        # Validate YAML structure
        if not isinstance(data, dict):
            LOG.error("YAML file must contain a dictionary at root level")
            raise ValueError("Invalid YAML structure: expected dictionary at root")
        
        # Check version compatibility
        if 'version' in data:
            try:
                if not are_versions_compatible(data['version'], JOBS_VERSION):
                    LOG.error(f"Jobs file version {data['version']} is not compatible with supported version {JOBS_VERSION}")
                    raise ValueError(f"Incompatible jobs file version: {data['version']} (supported: {JOBS_VERSION})")
            except ValueError as e:
                LOG.error(f"Version check failed: {e}")
                raise
        else:
            LOG.warning(f"No version found in {file_path}")
        
        if 'jobs' not in data:
            LOG.error("YAML file must contain 'jobs' key")
            raise ValueError("Invalid YAML structure: missing 'jobs' key")
        
        if not isinstance(data['jobs'], list):
            LOG.error("'jobs' key must contain a list")
            raise ValueError("Invalid YAML structure: 'jobs' must be a list")
        
        # Parse jobs
        for job_index, job_data in enumerate(data['jobs']):
            try:
                # Validate required fields
                if 'name' not in job_data:
                    raise ValueError(f"Job {job_index} missing required field: 'name'")
                if 'scenarios' not in job_data:
                    raise ValueError(f"Job {job_index} missing required field: 'scenarios'")
                
                # Parse scenarios (can be a list of integers or "all")
                scenarios = job_data['scenarios']
                if isinstance(scenarios, str) and scenarios.lower() == 'all':
                    # Load all 220 scenarios (0-219)
                    scenarios = list(range(220))
                elif isinstance(scenarios, list):
                    # Validate that all scenarios are integers
                    scenarios = [int(s) for s in scenarios]
                else:
                    raise ValueError(f"Job {job_index}: 'scenarios' must be a list or 'all'")
                
                # Parse experiment_nature (defaults to adversarial)
                experiment_nature_str = job_data.get('experiment_nature', 'adversarial')
                try:
                    experiment_nature = ExperimentNature.from_string(experiment_nature_str)
                except ValueError as e:
                    raise ValueError(f"Job {job_index}: {e}")
                
                # Create Job instance with auto-assigned ID
                job = Job(
                    id=job_index,
                    name=job_data['name'],
                    simulator=job_data.get('simulator', ''),
                    scenarios=scenarios,
                    ground_truth_count=job_data.get('ground_truth_count', 1),
                    ground_truth_cache_dir=job_data.get('ground_truth_cache_dir', ''),
                    filter=job_data.get('filter', ''),
                    generator=job_data.get('generator', ''),
                    model_instances=job_data.get('model_instances', []),
                    experiment_nature=experiment_nature,
                    instruction_generation_count=job_data.get('instruction_generation_count', 1),
                    instruction_count_per_generation=job_data.get('instruction_count_per_generation', 1),
                )
                
                self.jobs.append(job)
                LOG.debug(f"Loaded job {job.id}: '{job.name}' with {len(job.scenarios)} scenarios")
                
            except (ValueError, TypeError) as e:
                LOG.error(f"Failed to parse job {job_index}: {e}")
                raise ValueError(f"Invalid job configuration at index {job_index}: {e}")
        
        LOG.info(f"Successfully loaded {len(self.jobs)} jobs from {file_path}")
    
    def get_jobs(self) -> List[Job]:
        """
        Get the list of loaded jobs.
        
        Returns:
            List of Job objects
        """
        return self.jobs
    
    def __len__(self) -> int:
        """
        Get the number of loaded jobs.
        
        Returns:
            Number of jobs
        """
        return len(self.jobs)
    
    def is_ground_truth_cached(self, job_id: int, scenario_id: int, run_id: int) -> bool:
        """
        Check if ground truth logs for a specific scenario and run are already cached.
        
        Args:
            job_id: The job ID
            scenario_id: The scenario ID
            run_id: The run ID (0-based)
            
        Returns:
            True if cached logs exist, False otherwise
        """
        # Return True if cache directory exists
        cache_dir = self.get_ground_truth_cache_dir(job_id, scenario_id, run_id)
        return bool(cache_dir)
        
    def get_ground_truth_cache_dir(self, job_id: int, scenario_id: int, run_id: int) -> Optional[Path]:
        """
        Get the cache directory for a specific ground truth run if it exists.
        
        Args:
            job_id: The job ID
            scenario_id: The scenario ID
            run_id: The run ID

        Returns:
            The ground truth cache directory path, or empty string if not set
        """
        # Find the job with the given ID
        job = None
        for j in self.jobs:
            if j.id == job_id:
                job = j
                break
        
        if not job:
            LOG.warning(f"Job {job_id} not found")
            return ""
        
        # If no cache directory is specified, return None
        if not job.ground_truth_cache_dir:
            return None
        
        # Build the expected cache path: ${cache_dir}/sce_XXX/run_YYY
        cache_path = Path(job.ground_truth_cache_dir) / f"sce_{scenario_id:03d}" / f"run_{run_id:03d}"
        
        # Return cache path if the directory exists
        exists = cache_path.exists() and cache_path.is_dir()
        return cache_path if exists else None
