import pytest
import logging
from pathlib import Path
import yaml
import os

from untitled.core.settings import load_config
from untitled.core.job_handler import JobHandler
from untitled.core.version import CONFIG_VERSION, HOSTS_VERSION, JOBS_VERSION, are_versions_compatible


# File paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
JOBS_PATH = PROJECT_ROOT / "jobs.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def get_hosts_path_from_env() -> Path | None:
    """Read HOSTS_PATH from .env file."""
    if not ENV_PATH.exists():
        return None
    with open(ENV_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                if key.strip() == 'HOSTS_PATH':
                    path_str = value.strip()
                    if path_str:
                        return Path(path_str)
    return None

@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger("test_config_files")


class TestConfigFilesIntegration:
    """Integration tests that validate actual project configuration files."""

    def test_config_yaml_exists(self):
        """Test that config.yaml exists in the project root."""
        assert CONFIG_PATH.exists(), f"config.yaml not found at {CONFIG_PATH}"

    def test_config_yaml_valid_structure(self):
        """Test that config.yaml has valid YAML structure."""
        with open(CONFIG_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        assert isinstance(data, dict), "config.yaml must contain a dictionary"
        assert 'version' in data, "config.yaml must have a 'version' field"
        assert 'generators' in data, "config.yaml must have a 'generators' section"
        assert 'simulators' in data, "config.yaml must have a 'simulators' section"
        assert 'model_instances' in data, "config.yaml must have a 'model_instances' section"
        assert 'filters' in data, "config.yaml must have a 'filters' section"

    def test_config_yaml_version_compatible(self):
        """Test that config.yaml version is compatible with supported version."""
        with open(CONFIG_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        assert are_versions_compatible(data['version'], CONFIG_VERSION), \
            f"config.yaml version {data['version']} is not compatible with {CONFIG_VERSION}"

    def test_hosts_yaml_exists_or_not(self):
        """Test hosts.yaml existence (optional file)."""
        # This test just checks if we can determine existence - both outcomes are valid
        hosts_path = get_hosts_path_from_env()
        if hosts_path is None:
            pytest.skip("HOSTS_PATH not configured in .env (optional)")
        exists = hosts_path.exists()
        assert isinstance(exists, bool), "Should be able to check hosts.yaml existence"

    def test_hosts_yaml_valid_structure_if_exists(self):
        """Test that hosts.yaml has valid structure if it exists."""
        hosts_path = get_hosts_path_from_env()
        if hosts_path is None or not hosts_path.exists():
            pytest.skip("hosts.yaml does not exist (optional file)")
        
        with open(hosts_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert isinstance(data, dict), "hosts.yaml must contain a dictionary"
        assert 'version' in data, "hosts.yaml must have a 'version' field"
        assert 'hosts' in data, "hosts.yaml must have a 'hosts' section"
        assert isinstance(data['hosts'], list), "hosts must be a list"

    def test_hosts_yaml_version_compatible_if_exists(self):
        """Test that hosts.yaml version is compatible if file exists."""
        hosts_path = get_hosts_path_from_env()
        if hosts_path is None or not hosts_path.exists():
            pytest.skip("hosts.yaml does not exist (optional file)")
        
        with open(hosts_path, 'r') as f:
            data = yaml.safe_load(f)
        
        assert are_versions_compatible(data['version'], HOSTS_VERSION), \
            f"hosts.yaml version {data['version']} is not compatible with {HOSTS_VERSION}"

    def test_hosts_yaml_hosts_structure_if_exists(self):
        """Test that each host in hosts.yaml has required fields."""
        hosts_path = get_hosts_path_from_env()
        if hosts_path is None or not hosts_path.exists():
            pytest.skip("hosts.yaml does not exist (optional file)")
        
        with open(hosts_path, 'r') as f:
            data = yaml.safe_load(f)
        
        for idx, host in enumerate(data['hosts']):
            assert 'id' in host, f"Host {idx} missing 'id' field"
            assert 'host' in host, f"Host {idx} missing 'host' field"
            assert 'api_key' in host, f"Host {idx} missing 'api_key' field"

    def test_jobs_yaml_exists(self):
        """Test that jobs.yaml exists in the project root."""
        assert JOBS_PATH.exists(), f"jobs.yaml not found at {JOBS_PATH}"

    def test_jobs_yaml_valid_structure(self):
        """Test that jobs.yaml has valid YAML structure."""        
        with open(JOBS_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        assert isinstance(data, dict), "jobs.yaml must contain a dictionary"
        assert 'version' in data, "jobs.yaml must have a 'version' field"
        assert 'jobs' in data, "jobs.yaml must have a 'jobs' section"
        assert isinstance(data['jobs'], list), "jobs must be a list"

    def test_jobs_yaml_version_compatible(self):
        """Test that jobs.yaml version is compatible with supported version."""       
        with open(JOBS_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        assert are_versions_compatible(data['version'], JOBS_VERSION), \
            f"jobs.yaml version {data['version']} is not compatible with {JOBS_VERSION}"

    def test_jobs_yaml_jobs_structure(self):
        """Test that each job in jobs.yaml has required fields."""        
        with open(JOBS_PATH, 'r') as f:
            data = yaml.safe_load(f)
        
        for idx, job in enumerate(data['jobs']):
            assert 'name' in job, f"Job {idx} missing 'name' field"
            assert 'scenarios' in job, f"Job {idx} missing 'scenarios' field"
            
            # scenarios should be either a list or "all"
            scenarios = job['scenarios']
            assert isinstance(scenarios, (list, str)), \
                f"Job {idx}: 'scenarios' must be a list or string"
            
            if isinstance(scenarios, str):
                assert scenarios.lower() == 'all', \
                    f"Job {idx}: 'scenarios' string must be 'all'"

    def test_env_file_exists(self):
        """Test that .env file exists in the project root."""
        assert ENV_PATH.exists(), f".env file not found at {ENV_PATH}"

    def test_env_file_has_required_variables(self):
        """Test that .env file contains required environment variables."""        
        # Read .env file
        env_vars = {}
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        # Check for required variables
        required_vars = ['CARLA_PATH', 'CONDA_PATH', 'OUTPUT_PATH']
        for var in required_vars:
            assert var in env_vars, f"Required environment variable '{var}' not found in .env"
            assert env_vars[var], f"Environment variable '{var}' is empty in .env"

    def test_env_file_has_optional_variables(self):
        """Test that .env file contains optional environment variables (info only)."""
        # Read .env file
        env_vars = {}
        with open(ENV_PATH, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        # Check for optional variables - just skip if not present
        optional_vars = ['HOSTS_PATH']
        for var in optional_vars:
            if var not in env_vars or not env_vars[var]:
                pytest.skip(f"Optional environment variable '{var}' not configured in .env")

    def test_load_config_integration(self, logger):
        """Test that load_config successfully loads the actual project config."""
        try:
            settings = load_config(PROJECT_ROOT, logger)
            
            # Verify basic structure (lists of configs)
            assert hasattr(settings, 'filters')
            assert hasattr(settings, 'generators')
            assert hasattr(settings, 'simulators')
            assert hasattr(settings, 'model_instances')
            assert hasattr(settings, 'hosts')
            
            # Verify filters
            for filter_config in settings.filters:
                assert filter_config.id
                assert filter_config.max_frames_per_minute >= 0
            
            # Verify generators
            for generator in settings.generators:
                assert generator.id
                assert generator.design in ['single_step_vlm', 'multi_step_vlm']
            
            # Verify simulators
            for simulator in settings.simulators:
                assert simulator.id
                assert simulator.type in ['carla']
            
            # Verify model instances
            for instance in settings.model_instances:
                assert instance.id
                assert instance.model_name
                assert instance.type in ["vlm", "llm"]
                assert instance.priority is not None and instance.priority >= 0
                assert instance.batch_size is not None and instance.batch_size > 0
                # Each instance should have either local_config or remote_config
                assert (instance.local_config is not None) or (instance.remote_config is not None), \
                    f"Model instance {instance.id} must have either local_config or remote_config"
                assert not (instance.local_config is not None and instance.remote_config is not None), \
                    f"Model instance {instance.id} cannot have both local_config and remote_config"
            
            # Verify hosts (if any)
            for host in settings.hosts:
                assert host.id
                assert host.host
                assert host.api_key is not None
            
        except Exception as e:
            pytest.fail(f"Failed to load project config: {e}")

    def test_load_jobs_integration(self):
        """Test that JobHandler successfully loads the actual jobs.yaml."""        
        try:
            job_handler = JobHandler(str(JOBS_PATH))
            
            # Verify jobs were loaded
            assert len(job_handler.jobs) > 0, "No jobs loaded from jobs.yaml"
            
            # Verify each job has valid structure
            for job in job_handler.jobs:
                assert job.id is not None
                assert job.name
                assert job.simulator, f"Job '{job.name}' missing simulator"
                assert isinstance(job.scenarios, list)
                assert len(job.scenarios) > 0, f"Job '{job.name}' has no scenarios"
                assert job.ground_truth_count >= 0
                assert job.instruction_generation_count >= 0
                assert job.instruction_count_per_generation >= 0
                # Optional fields (can be empty strings or empty lists)
                assert isinstance(job.filter, str)
                assert isinstance(job.generator, str)
                assert isinstance(job.model_instances, list)
                
        except Exception as e:
            pytest.fail(f"Failed to load jobs.yaml: {e}")

    def test_model_instances_reference_valid_hosts(self, logger):
        """Test that remote model instances reference hosts that exist in hosts.yaml."""
        # Load config
        settings = load_config(PROJECT_ROOT, logger)
        
        # Get all remote instances
        remote_instances = [i for i in settings.model_instances if i.remote_config is not None]
        
        # Skip test if no remote instances are configured
        if not remote_instances:
            pytest.skip("No remote model instances configured")
        
        # Get all host IDs
        host_ids = {host.id for host in settings.hosts}
        
        # Check that hosts.yaml exists and has hosts if remote instances are configured
        if not host_ids:
            pytest.fail(f"Remote model instances are configured but no hosts are defined in hosts.yaml. "
                       f"Remote instances: {[i.id for i in remote_instances]}")
        
        # Check remote model instances
        for instance in remote_instances:
            assert instance.remote_config.host_id in host_ids, \
                f"Model instance '{instance.id}' references non-existent host '{instance.remote_config.host_id}'. " \
                f"Available hosts: {list(host_ids)}"

    def test_config_host_ids_exist_in_hosts_yaml(self):
        """Test that host IDs referenced in config.yaml exist in hosts.yaml (direct file comparison)."""
        # Read config.yaml directly
        with open(CONFIG_PATH, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Get all host IDs referenced in config.yaml remote_config sections
        referenced_host_ids = set()
        for instance in config_data.get('model_instances', []):
            remote_config = instance.get('remote_config')
            if remote_config and 'host_id' in remote_config:
                referenced_host_ids.add(remote_config['host_id'])
        
        # Skip if no remote instances reference hosts
        if not referenced_host_ids:
            pytest.skip("No host IDs referenced in config.yaml")
        
        # Read hosts.yaml
        hosts_path = get_hosts_path_from_env()
        if hosts_path is None or not hosts_path.exists():
            pytest.fail(f"config.yaml references host IDs {referenced_host_ids} but hosts.yaml is not configured")
        
        with open(hosts_path, 'r') as f:
            hosts_data = yaml.safe_load(f)
        
        # Get all defined host IDs
        defined_host_ids = {host['id'] for host in hosts_data.get('hosts', [])}
        
        # Check that all referenced host IDs exist
        missing_hosts = referenced_host_ids - defined_host_ids
        assert not missing_hosts, \
            f"config.yaml references host IDs not defined in hosts.yaml: {missing_hosts}. " \
            f"Available hosts: {defined_host_ids}"

    def test_generator_steps_reference_valid_models(self, logger):
        """Test that generator steps reference model names that exist in model_instances."""
        # Load config
        settings = load_config(PROJECT_ROOT, logger)
        
        # Get all model names
        model_names = {instance.model_name for instance in settings.model_instances}
        
        # Check all generators' steps
        for generator in settings.generators:
            for step in generator.steps:
                assert step.model_name in model_names, \
                    f"Generator '{generator.id}' step '{step.id}' references non-existent model '{step.model_name}'"

    def test_model_instances_step_ids_reference_valid_steps(self, logger):
        """Test that model instances reference step IDs that exist in generator steps."""
        # Load config
        settings = load_config(PROJECT_ROOT, logger)
        
        # Get all step IDs from all generators
        step_ids = set()
        for generator in settings.generators:
            for step in generator.steps:
                step_ids.add(step.id)
        
        # Check model instances
        for instance in settings.model_instances:
            for step_id in instance.step_ids:
                assert step_id in step_ids, \
                    f"Model instance {instance.id} references non-existent step '{step_id}'"

    def test_jobs_reference_valid_config_ids(self, logger):
        """Test that jobs reference valid IDs from config.yaml."""
        # Load config
        settings = load_config(PROJECT_ROOT, logger)
        
        # Get all valid IDs from config
        simulator_ids = {s.id for s in settings.simulators}
        filter_ids = {f.id for f in settings.filters}
        generator_ids = {g.id for g in settings.generators}
        model_instance_ids = {m.id for m in settings.model_instances}
        
        # Load jobs
        job_handler = JobHandler(str(JOBS_PATH))
        
        # Check each job references valid config IDs
        for job in job_handler.jobs:
            # Simulator is required
            assert job.simulator in simulator_ids, \
                f"Job '{job.name}' references non-existent simulator '{job.simulator}'. " \
                f"Available: {simulator_ids}"
            
            # Filter is optional but must be valid if specified
            if job.filter:
                assert job.filter in filter_ids, \
                    f"Job '{job.name}' references non-existent filter '{job.filter}'. " \
                    f"Available: {filter_ids}"
            
            # Generator is optional but must be valid if specified
            if job.generator:
                assert job.generator in generator_ids, \
                    f"Job '{job.name}' references non-existent generator '{job.generator}'. " \
                    f"Available: {generator_ids}"
            
            # Model instances are optional but must be valid if specified
            for instance_id in job.model_instances:
                assert instance_id in model_instance_ids, \
                    f"Job '{job.name}' references non-existent model instance '{instance_id}'. " \
                    f"Available: {model_instance_ids}"
