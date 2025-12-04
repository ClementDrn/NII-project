from untitled.core.base import LOG
from untitled.core.version import JOBS_VERSION, are_versions_compatible

from dataclasses import dataclass
from pathlib import Path
from typing import List
import yaml


@dataclass
class Job:
    """
    Represents a job containing scenarios to run with specific settings.
    
    Attributes:
        id: Unique identifier for the job (auto-assigned)
        name: Human-readable name for the job
        scenarios: List of scenario IDs to run
        ground_truth_count: Number of baseline runs without instructions
        experiment_count: Number of runs with VLM-generated instructions
    """
    id: int
    name: str
    scenarios: List[int]
    ground_truth_count: int = 1
    experiment_count: int = 1


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
        
        The YAML file should have the following structure:
        ```yaml
        version: "0.0"
        jobs:
          - name: "Job Name"
            scenarios: [1, 2, 3]
            ground_truth_count: 1
            experiment_count: 5
        ```
        
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
            if not are_versions_compatible(data['version'], JOBS_VERSION):
                LOG.error(f"Jobs file version {data['version']} is not compatible with supported version {JOBS_VERSION}")
                raise ValueError(f"Incompatible jobs file version: {data['version']} (supported: {JOBS_VERSION})")
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
                
                # Create Job instance with auto-assigned ID
                job = Job(
                    id=job_index,
                    name=job_data['name'],
                    scenarios=scenarios,
                    ground_truth_count=job_data.get('ground_truth_count', 1),
                    experiment_count=job_data.get('experiment_count', 1),
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
        
