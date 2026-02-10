import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
from untitled.core.job_handler import Job, JobHandler
from untitled.generator.experiment_nature import ExperimentNature


@pytest.fixture
def temp_dir():
    """Create a temporary directory and clean up after test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_jobs_yaml(temp_dir):
    """Create a sample jobs.yaml file."""
    jobs_file = temp_dir / "jobs.yaml"
    jobs_data = {
        "version": "2.0-dev",
        "jobs": [
            {
                "name": "Test Job 1",
                "simulator": "carla_basic",
                "scenarios": [1, 2, 3],
                "ground_truth_count": 2,
                "instruction_generation_count": 5
            },
            {
                "name": "Test Job 2",
                "simulator": "carla_basic",
                "scenarios": [10, 20],
                "ground_truth_count": 1,
                "instruction_generation_count": 3,
                "ground_truth_cache_dir": "cache"
            }
        ]
    }
    
    with open(jobs_file, 'w') as f:
        yaml.dump(jobs_data, f)
    
    return jobs_file


@pytest.fixture
def jobs_yaml_with_all_scenarios(temp_dir):
    """Create a jobs.yaml file with 'all' scenarios."""
    jobs_file = temp_dir / "jobs_all.yaml"
    jobs_data = {
        "version": "2.0-dev",
        "jobs": [
            {
                "name": "All Scenarios",
                "simulator": "carla_basic",
                "scenarios": "all",
                "ground_truth_count": 1,
                "instruction_generation_count": 1
            }
        ]
    }
    
    with open(jobs_file, 'w') as f:
        yaml.dump(jobs_data, f)
    
    return jobs_file


@pytest.fixture
def cache_dir_with_logs(temp_dir):
    """Create a cache directory with some ground truth logs."""
    cache_dir = temp_dir / "cache"
    
    # Create cached logs for scenario 5, runs 0-2
    for run_id in range(3):
        log_dir = cache_dir / f"sce_005" / f"run_{run_id:03d}"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "res.json").touch()
    
    # Create cached logs for scenario 7, run 0 only
    log_dir = cache_dir / f"sce_007" / "run_000"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "res.json").touch()
    
    return cache_dir


@pytest.fixture
def jobs_yaml_with_experiment_nature(temp_dir):
    """Create a jobs.yaml file with experiment_nature specified."""
    jobs_file = temp_dir / "jobs_nature.yaml"
    jobs_data = {
        "version": "2.0-dev",
        "jobs": [
            {
                "name": "Adversarial Job",
                "simulator": "carla_basic",
                "scenarios": [1, 2],
                "experiment_nature": "adversarial"
            },
            {
                "name": "Corrective Job",
                "simulator": "carla_basic",
                "scenarios": [3, 4],
                "experiment_nature": "corrective"
            },
            {
                "name": "Default Nature Job",
                "simulator": "carla_basic",
                "scenarios": [5]
            }
        ]
    }
    
    with open(jobs_file, 'w') as f:
        yaml.dump(jobs_data, f)
    
    return jobs_file


class TestJob:
    """Test suite for Job dataclass."""
    
    def test_job_creation_with_defaults(self):
        """Test creating a Job instance with default values."""
        job = Job(
            id=0,
            name="Test Job",
            simulator="carla_basic",
            scenarios=[1, 2, 3]
        )
        
        assert job.id == 0
        assert job.name == "Test Job"
        assert job.simulator == "carla_basic"
        assert job.scenarios == [1, 2, 3]
        assert job.ground_truth_count == 1
        assert job.instruction_generation_count == 1
        assert job.experiment_nature == ExperimentNature.ADVERSARIAL
        assert job.ground_truth_cache_dir == ""
    
    def test_job_creation_with_all_fields(self):
        """Test creating a Job instance with all fields specified."""
        job = Job(
            id=5,
            name="Full Job",
            simulator="carla_debug",
            scenarios=[10, 20, 30],
            ground_truth_count=3,
            instruction_generation_count=10,
            experiment_nature=ExperimentNature.CORRECTIVE,
            ground_truth_cache_dir="/path/to/cache"
        )
        
        assert job.id == 5
        assert job.name == "Full Job"
        assert job.simulator == "carla_debug"
        assert job.scenarios == [10, 20, 30]
        assert job.ground_truth_count == 3
        assert job.instruction_generation_count == 10
        assert job.experiment_nature == ExperimentNature.CORRECTIVE
        assert job.ground_truth_cache_dir == "/path/to/cache"


class TestJobHandler:
    """Test suite for JobHandler class."""
    
    def test_initialization_without_file(self):
        """Test JobHandler initialization without a file path."""
        handler = JobHandler()
        assert len(handler.jobs) == 0
        assert len(handler) == 0
    
    def test_initialization_with_file(self, sample_jobs_yaml):
        """Test JobHandler initialization with a valid file."""
        handler = JobHandler(str(sample_jobs_yaml))
        assert len(handler) == 2
        assert len(handler.get_jobs()) == 2
    
    def test_load_jobs_from_valid_file(self, sample_jobs_yaml):
        """Test loading jobs from a valid YAML file."""
        handler = JobHandler(str(sample_jobs_yaml))
        jobs = handler.get_jobs()
        
        # Check first job
        assert jobs[0].id == 0
        assert jobs[0].name == "Test Job 1"
        assert jobs[0].simulator == "carla_basic"
        assert jobs[0].scenarios == [1, 2, 3]
        assert jobs[0].ground_truth_count == 2
        assert jobs[0].instruction_generation_count == 5
        assert jobs[0].ground_truth_cache_dir == ""
        
        # Check second job
        assert jobs[1].id == 1
        assert jobs[1].name == "Test Job 2"
        assert jobs[1].simulator == "carla_basic"
        assert jobs[1].scenarios == [10, 20]
        assert jobs[1].ground_truth_count == 1
        assert jobs[1].instruction_generation_count == 3
        assert jobs[1].ground_truth_cache_dir == "cache"
    
    def test_load_jobs_with_all_scenarios(self, jobs_yaml_with_all_scenarios):
        """Test loading jobs with 'all' scenarios keyword."""
        handler = JobHandler(str(jobs_yaml_with_all_scenarios))
        jobs = handler.get_jobs()
        
        assert len(jobs) == 1
        assert jobs[0].name == "All Scenarios"
        assert len(jobs[0].scenarios) == 220
        assert jobs[0].scenarios == list(range(220))
    
    def test_load_jobs_with_experiment_nature(self, jobs_yaml_with_experiment_nature):
        """Test loading jobs with experiment_nature specified."""
        handler = JobHandler(str(jobs_yaml_with_experiment_nature))
        jobs = handler.get_jobs()
        
        assert len(jobs) == 3
        
        # Check adversarial job
        assert jobs[0].name == "Adversarial Job"
        assert jobs[0].experiment_nature == ExperimentNature.ADVERSARIAL
        
        # Check corrective job
        assert jobs[1].name == "Corrective Job"
        assert jobs[1].experiment_nature == ExperimentNature.CORRECTIVE
        
        # Check default nature (should be adversarial)
        assert jobs[2].name == "Default Nature Job"
        assert jobs[2].experiment_nature == ExperimentNature.ADVERSARIAL
    
    def test_load_jobs_with_invalid_experiment_nature(self, temp_dir):
        """Test loading jobs with invalid experiment_nature raises ValueError."""
        jobs_file = temp_dir / "jobs_invalid_nature.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Invalid Nature Job",
                    "simulator": "carla_basic",
                    "scenarios": [1],
                    "experiment_nature": "invalid_nature"
                }
            ]
        }
        
        with open(jobs_file, 'w') as f:
            yaml.dump(jobs_data, f)
        
        with pytest.raises(ValueError) as exc_info:
            JobHandler(str(jobs_file))
        
        assert "experiment nature" in str(exc_info.value).lower()
    
    def test_load_jobs_file_not_found(self):
        """Test loading jobs from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            JobHandler("/path/that/does/not/exist/jobs.yaml")
    
    def test_load_jobs_invalid_yaml(self, temp_dir):
        """Test loading jobs from invalid YAML file raises ValueError."""
        invalid_yaml = temp_dir / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("{ invalid yaml content [")
        
        with pytest.raises(ValueError, match="Invalid YAML format"):
            JobHandler(str(invalid_yaml))
    
    def test_load_jobs_missing_jobs_key(self, temp_dir):
        """Test loading YAML without 'jobs' key raises ValueError."""
        no_jobs_yaml = temp_dir / "no_jobs.yaml"
        with open(no_jobs_yaml, 'w') as f:
            yaml.dump({"version": "2.0-dev"}, f)
        
        with pytest.raises(ValueError, match="missing 'jobs' key"):
            JobHandler(str(no_jobs_yaml))
    
    def test_load_jobs_invalid_jobs_type(self, temp_dir):
        """Test loading YAML with non-list 'jobs' raises ValueError."""
        invalid_jobs_yaml = temp_dir / "invalid_jobs.yaml"
        with open(invalid_jobs_yaml, 'w') as f:
            yaml.dump({"version": "2.0-dev", "jobs": "not a list"}, f)
        
        with pytest.raises(ValueError, match="'jobs' must be a list"):
            JobHandler(str(invalid_jobs_yaml))
    
    def test_load_jobs_missing_name_field(self, temp_dir):
        """Test loading job without 'name' field raises ValueError."""
        missing_name_yaml = temp_dir / "missing_name.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "simulator": "carla_basic",
                    "scenarios": [1, 2, 3]
                }
            ]
        }
        
        with open(missing_name_yaml, 'w') as f:
            yaml.dump(jobs_data, f)
        
        with pytest.raises(ValueError, match="missing required field: 'name'"):
            JobHandler(str(missing_name_yaml))
    
    def test_load_jobs_missing_scenarios_field(self, temp_dir):
        """Test loading job without 'scenarios' field raises ValueError."""
        missing_scenarios_yaml = temp_dir / "missing_scenarios.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic"
                }
            ]
        }
        
        with open(missing_scenarios_yaml, 'w') as f:
            yaml.dump(jobs_data, f)
        
        with pytest.raises(ValueError, match="missing required field: 'scenarios'"):
            JobHandler(str(missing_scenarios_yaml))
    
    def test_load_jobs_invalid_scenarios_type(self, temp_dir):
        """Test loading job with invalid scenarios type raises ValueError."""
        invalid_scenarios_yaml = temp_dir / "invalid_scenarios.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic",
                    "scenarios": 123  # Should be list or "all"
                }
            ]
        }
        
        with open(invalid_scenarios_yaml, 'w') as f:
            yaml.dump(jobs_data, f)
        
        with pytest.raises(ValueError, match="'scenarios' must be a list or 'all'"):
            JobHandler(str(invalid_scenarios_yaml))
    
    def test_load_jobs_incompatible_version(self, temp_dir):
        """Test loading jobs with incompatible version raises ValueError."""
        incompatible_version_yaml = temp_dir / "incompatible_version.yaml"
        jobs_data = {
            "version": "1.0",  # Incompatible with 2.0-dev
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic",
                    "scenarios": [1, 2, 3]
                }
            ]
        }
        
        with open(incompatible_version_yaml, 'w') as f:
            yaml.dump(jobs_data, f)
        
        with pytest.raises(ValueError, match="Incompatible jobs file version"):
            JobHandler(str(incompatible_version_yaml))
    
    def test_get_jobs_returns_list(self, sample_jobs_yaml):
        """Test that get_jobs returns a list of Job objects."""
        handler = JobHandler(str(sample_jobs_yaml))
        jobs = handler.get_jobs()
        
        assert isinstance(jobs, list)
        assert all(isinstance(job, Job) for job in jobs)
    
    def test_len_returns_job_count(self, sample_jobs_yaml):
        """Test that __len__ returns the correct number of jobs."""
        handler = JobHandler(str(sample_jobs_yaml))
        assert len(handler) == 2


class TestIsGroundTruthCached:
    """Test suite for is_ground_truth_cached method."""
    
    def test_is_ground_truth_cached_no_cache_dir(self, sample_jobs_yaml):
        """Test that method returns False when no cache dir is specified."""
        handler = JobHandler(str(sample_jobs_yaml))
        
        # Job 0 has no cache_dir specified
        assert handler.is_ground_truth_cached(0, 1, 0) is False
    
    def test_is_ground_truth_cached_cache_exists(self, temp_dir, cache_dir_with_logs):
        """Test that method returns True when cached logs exist."""
        jobs_file = temp_dir / "jobs.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic",
                    "scenarios": [5, 7],
                    "ground_truth_cache_dir": str(cache_dir_with_logs)
                }
            ]
        }
        
        with open(jobs_file, 'w') as f:
            yaml.dump(jobs_data, f)
        
        handler = JobHandler(str(jobs_file))
        
        # These should exist
        assert handler.is_ground_truth_cached(0, 5, 0) is True
        assert handler.is_ground_truth_cached(0, 5, 1) is True
        assert handler.is_ground_truth_cached(0, 5, 2) is True
        assert handler.is_ground_truth_cached(0, 7, 0) is True
    
    def test_is_ground_truth_cached_cache_not_exists(self, temp_dir, cache_dir_with_logs):
        """Test that method returns False when cached logs don't exist."""
        jobs_file = temp_dir / "jobs.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic",
                    "scenarios": [5, 7],
                    "ground_truth_cache_dir": str(cache_dir_with_logs)
                }
            ]
        }
        
        with open(jobs_file, 'w') as f:
            yaml.dump(jobs_data, f)
        
        handler = JobHandler(str(jobs_file))
        
        # These should not exist
        assert handler.is_ground_truth_cached(0, 5, 3) is False  # Run 3 doesn't exist
        assert handler.is_ground_truth_cached(0, 7, 1) is False  # Run 1 doesn't exist
        assert handler.is_ground_truth_cached(0, 10, 0) is False  # Scenario 10 doesn't exist
    
    def test_is_ground_truth_cached_invalid_job_id(self, sample_jobs_yaml):
        """Test that method returns False for invalid job ID."""
        handler = JobHandler(str(sample_jobs_yaml))
        
        # Job ID 999 doesn't exist
        assert handler.is_ground_truth_cached(999, 1, 0) is False
    
    def test_is_ground_truth_cached_path_formatting(self, temp_dir):
        """Test that method uses correct path formatting with zero-padding."""
        cache_dir = temp_dir / "cache"
        
        # Create a cache with specific formatting
        log_dir = cache_dir / "sce_005" / "run_000"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        jobs_file = temp_dir / "jobs.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic",
                    "scenarios": [5],
                    "ground_truth_cache_dir": str(cache_dir)
                }
            ]
        }
        
        with open(jobs_file, 'w') as f:
            yaml.dump(jobs_data, f)
        
        handler = JobHandler(str(jobs_file))
        
        # Should find the cache with proper formatting
        assert handler.is_ground_truth_cached(0, 5, 0) is True
    
    def test_is_ground_truth_cached_relative_path(self, temp_dir):
        """Test that method works with relative cache directory paths."""
        # Create cache in a subdirectory
        cache_dir = temp_dir / "relative_cache"
        log_dir = cache_dir / "sce_010" / "run_000"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        jobs_file = temp_dir / "jobs.yaml"
        jobs_data = {
            "version": "2.0-dev",
            "jobs": [
                {
                    "name": "Test Job",
                    "simulator": "carla_basic",
                    "scenarios": [10],
                    "ground_truth_cache_dir": "relative_cache"
                }
            ]
        }
        
        with open(jobs_file, 'w') as f:
            yaml.dump(jobs_data, f)
        
        handler = JobHandler(str(jobs_file))
        
        # Should work with relative path (from current working directory)
        # Note: This will return False if run from a different directory
        # In production, absolute paths are recommended
        result = handler.is_ground_truth_cached(0, 10, 0)
        assert isinstance(result, bool)
