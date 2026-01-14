import pytest
import tempfile
import shutil
from pathlib import Path
import yaml
import logging

from untitled.core.settings import (
    Settings,
    GeneratorConfig,
    CARLAConfig,
    ModelInstanceConfig,
    LocalVLMConfig,
    RemoteVLMConfig,
    HostConfig,
    GeneratorStepConfig,
    FilterConfig,
    load_config
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config files and clean up after test."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def logger():
    """Create a logger for testing."""
    return logging.getLogger("test_settings")


class TestLocalVLMConfig:
    """Test suite for LocalVLMConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = LocalVLMConfig()
        
        assert config.gpu_id == 0
        assert config.is_parallel_to_simulator is False
        assert config.tensor_parallel_size == 1
        assert config.gpu_memory_utilization == 0.95
        assert config.max_num_seqs == 64
        assert config.enable_chunked_prefill is True
        assert config.enable_prefix_caching is False
        assert config.dtype == "bfloat16"

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = LocalVLMConfig(
            gpu_id=1,
            is_parallel_to_simulator=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            max_num_seqs=32,
            enable_chunked_prefill=False,
            enable_prefix_caching=True,
            dtype="float16"
        )
        
        assert config.gpu_id == 1
        assert config.is_parallel_to_simulator is True
        assert config.tensor_parallel_size == 2
        assert config.gpu_memory_utilization == 0.8
        assert config.max_num_seqs == 32
        assert config.enable_chunked_prefill is False
        assert config.enable_prefix_caching is True
        assert config.dtype == "float16"


class TestRemoteVLMConfig:
    """Test suite for RemoteVLMConfig."""

    def test_initialization(self):
        """Test that RemoteVLMConfig can be initialized."""
        config = RemoteVLMConfig(host_id="my_vlm_host")
        
        assert config.host_id == "my_vlm_host"


class TestModelInstanceConfig:
    """Test suite for ModelInstanceConfig."""

    def test_local_instance(self):
        """Test local model instance configuration."""
        local_config = LocalVLMConfig(gpu_id=0)
        instance = ModelInstanceConfig(
            id="r4b_0",
            model_name="r4b",
            type="vlm",
            priority=5,
            batch_size=4,
            step_ids=["description", "planning"],
            local_config=local_config
        )
        
        assert instance.id == "r4b_0"
        assert instance.model_name == "r4b"
        assert instance.type == "vlm"
        assert instance.priority == 5
        assert instance.batch_size == 4
        assert instance.step_ids == ["description", "planning"]
        assert instance.local_config is not None
        assert instance.local_config.gpu_id == 0
        assert instance.remote_config is None

    def test_remote_instance(self):
        """Test remote model instance configuration."""
        remote_config = RemoteVLMConfig(host_id="my_vlm_host")
        instance = ModelInstanceConfig(
            id="r4b_1",
            model_name="r4b",
            type="vlm",
            priority=3,
            batch_size=2,
            step_ids=["description"],
            remote_config=remote_config
        )
        
        assert instance.id == "r4b_1"
        assert instance.model_name == "r4b"
        assert instance.type == "vlm"
        assert instance.priority == 3
        assert instance.batch_size == 2
        assert instance.step_ids == ["description"]
        assert instance.remote_config is not None
        assert instance.remote_config.host_id == "my_vlm_host"
        assert instance.local_config is None


class TestGeneratorStepConfig:
    """Test suite for GeneratorStepConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        step = GeneratorStepConfig(
            id="description",
            model_name="r4b"
        )
        
        assert step.id == "description"
        assert step.model_name == "r4b"
        assert step.temperature == 0.7
        assert step.max_tokens == 512
        assert step.extra_body == {}

    def test_custom_values(self):
        """Test that custom values can be set."""
        step = GeneratorStepConfig(
            id="planning",
            model_name="r4b",
            temperature=0.0,
            max_tokens=1024,
            extra_body={"chat_template_kwargs": {"thinking_mode": "long"}}
        )
        
        assert step.id == "planning"
        assert step.model_name == "r4b"
        assert step.temperature == 0.0
        assert step.max_tokens == 1024
        assert step.extra_body == {"chat_template_kwargs": {"thinking_mode": "long"}}


class TestCARLAConfig:
    """Test suite for CARLAConfig."""

    def test_initialization_from_env(self, monkeypatch):
        """Test that CARLAConfig can be initialized from environment variables."""
        monkeypatch.setenv("CARLA_PATH", "/path/to/carla")
        monkeypatch.setenv("CONDA_PATH", "/path/to/conda")
        monkeypatch.setenv("OUTPUT_PATH", "/path/to/output")
        
        config = CARLAConfig()
        
        assert config.path == Path("/path/to/carla")
        assert config.conda_path == "/path/to/conda"
        assert config.output_path == Path("/path/to/output")
        assert config.debug is False
        assert config.gpu_id == 0

    def test_debug_flag(self, monkeypatch):
        """Test that debug flag can be set."""
        monkeypatch.setenv("CARLA_PATH", "/path/to/carla")
        monkeypatch.setenv("CONDA_PATH", "/path/to/conda")
        monkeypatch.setenv("OUTPUT_PATH", "/path/to/output")
        
        config = CARLAConfig(debug=True, gpu_id=1)
        
        assert config.debug is True
        assert config.gpu_id == 1


class TestGeneratorConfig:
    """Test suite for GeneratorConfig."""

    def test_default_design(self):
        """Test default generator design."""
        config = GeneratorConfig()
        
        assert config.design == "single_step_vlm"
        assert config.steps == []

    def test_multi_step_design(self):
        """Test multi-step generator configuration."""
        steps = [
            GeneratorStepConfig(id="description", model_name="r4b"),
            GeneratorStepConfig(id="planning", model_name="r4b")
        ]
        config = GeneratorConfig(design="multi_step_vlm", steps=steps)
        
        assert config.design == "multi_step_vlm"
        assert len(config.steps) == 2
        assert config.steps[0].id == "description"
        assert config.steps[1].id == "planning"


class TestHostConfig:
    """Test suite for HostConfig."""

    def test_initialization(self):
        """Test that HostConfig can be initialized."""
        host = HostConfig(
            id="my_vlm_host",
            host="http://localhost:8000",
            api_key="EMPTY"
        )
        
        assert host.id == "my_vlm_host"
        assert host.host == "http://localhost:8000"
        assert host.api_key == "EMPTY"


class TestSettings:
    """Test suite for Settings."""

    def test_minimal_settings(self, monkeypatch):
        """Test settings with minimal required configuration."""
        # Set required environment variables
        monkeypatch.setenv("CARLA_PATH", "/path/to/carla")
        monkeypatch.setenv("CONDA_PATH", "/path/to/conda")
        monkeypatch.setenv("OUTPUT_PATH", "/path/to/output")
        
        filter_config = FilterConfig()
        generator = GeneratorConfig()
        carla = CARLAConfig()
        settings = Settings(filter=filter_config, generator=generator, carla=carla)
        
        assert settings.generator.design == "single_step_vlm"
        assert settings.carla.debug is False
        assert settings.model_instances == []
        assert settings.hosts == []

    def test_full_settings(self, monkeypatch):
        """Test settings with full configuration."""
        # Set required environment variables
        monkeypatch.setenv("CARLA_PATH", "/path/to/carla")
        monkeypatch.setenv("CONDA_PATH", "/path/to/conda")
        monkeypatch.setenv("OUTPUT_PATH", "/path/to/output")
        
        # Create generator config
        steps = [
            GeneratorStepConfig(id="description", model_name="r4b"),
            GeneratorStepConfig(id="planning", model_name="r4b")
        ]
        generator = GeneratorConfig(design="multi_step_vlm", steps=steps)
        
        # Create CARLA config
        carla = CARLAConfig(debug=True, gpu_id=0)
        
        # Create model instances
        model_instances = [
            ModelInstanceConfig(
                id="r4b_0",
                model_name="r4b",
                type="vlm",
                step_ids=["description", "planning"],
                local_config=LocalVLMConfig(gpu_id=0)
            ),
            ModelInstanceConfig(
                id="r4b_1",
                model_name="r4b",
                type="vlm",
                step_ids=["description"],
                remote_config=RemoteVLMConfig(host_id="my_vlm_host")
            )
        ]
        
        # Create hosts
        hosts = [
            HostConfig(
                id="my_vlm_host",
                host="http://localhost:8000",
                api_key="EMPTY"
            )
        ]
        
        # Create settings
        settings = Settings(
            filter=FilterConfig(),
            generator=generator,
            carla=carla,
            model_instances=model_instances,
            hosts=hosts
        )
        
        assert settings.generator.design == "multi_step_vlm"
        assert len(settings.generator.steps) == 2
        assert settings.carla.debug is True
        assert len(settings.model_instances) == 2
        assert len(settings.hosts) == 1
        assert settings.model_instances[0].local_config is not None
        assert settings.model_instances[1].remote_config is not None
        assert settings.hosts[0].id == "my_vlm_host"


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_config_minimal(self, temp_config_dir, logger, monkeypatch):
        """Test loading minimal configuration."""
        # Create .env file
        env_content = """CARLA_PATH=/path/to/carla
CONDA_PATH=/path/to/conda
OUTPUT_PATH=/path/to/output
"""
        with open(temp_config_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create config.yaml
        config_data = {
            "version": "2.0-dev",
            "filter": {
                "max_frames_per_minute": 60,
                "clustering_threshold": 0.99,
                "max_cluster_count": 30
            },
            "generator": {
                "design": "single_step_vlm",
                "steps": []
            },
            "carla": {
                "debug": False,
                "gpu_id": 0
            },
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Load config
        settings = load_config(temp_config_dir, logger)
        
        assert settings.generator.design == "single_step_vlm"
        assert settings.carla.debug is False
        assert settings.model_instances == []
        assert settings.hosts == []

    def test_load_config_with_hosts(self, temp_config_dir, logger, monkeypatch):
        """Test loading configuration with hosts.yaml."""
        # Create .env file
        env_content = """CARLA_PATH=/path/to/carla
CONDA_PATH=/path/to/conda
OUTPUT_PATH=/path/to/output
"""
        with open(temp_config_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create config.yaml
        config_data = {
            "version": "2.0-dev",
            "filter": {
                "max_frames_per_minute": 60,
                "clustering_threshold": 0.99,
                "max_cluster_count": 30
            },
            "generator": {
                "design": "multi_step_vlm",
                "steps": [
                    {
                        "id": "description",
                        "model_name": "r4b",
                        "temperature": 0.0,
                        "max_tokens": 1024
                    }
                ]
            },
            "carla": {
                "debug": True,
                "gpu_id": 0
            },
            "model_instances": [
                {
                    "id": "r4b_0",
                    "model_name": "r4b",
                    "type": "vlm",
                    "step_ids": ["description"],
                    "local_config": {
                        "gpu_id": 0,
                        "is_parallel_to_simulator": False
                    }
                }
            ]
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Create hosts.yaml
        hosts_data = {
            "version": "1.0-dev",
            "hosts": [
                {
                    "id": "my_vlm_host",
                    "host": "http://localhost:8000",
                    "api_key": "EMPTY"
                }
            ]
        }
        with open(temp_config_dir / "hosts.yaml", "w") as f:
            yaml.dump(hosts_data, f)
        
        # Load config
        settings = load_config(temp_config_dir, logger)
        
        assert settings.generator.design == "multi_step_vlm"
        assert len(settings.generator.steps) == 1
        assert settings.generator.steps[0].id == "description"
        assert settings.carla.debug is True
        assert len(settings.model_instances) == 1
        assert settings.model_instances[0].id == "r4b_0"
        assert settings.model_instances[0].local_config.gpu_id == 0
        assert len(settings.hosts) == 1
        assert settings.hosts[0].id == "my_vlm_host"
        assert settings.hosts[0].host == "http://localhost:8000"

    def test_load_config_without_hosts_file(self, temp_config_dir, logger, monkeypatch):
        """Test loading configuration when hosts.yaml doesn't exist."""
        # Create .env file
        env_content = """CARLA_PATH=/path/to/carla
CONDA_PATH=/path/to/conda
OUTPUT_PATH=/path/to/output
"""
        with open(temp_config_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create config.yaml
        config_data = {
            "version": "2.0-dev",
            "filter": {
                "max_frames_per_minute": 60,
                "clustering_threshold": 0.99,
                "max_cluster_count": 30
            },
            "generator": {
                "design": "single_step_vlm",
                "steps": []
            },
            "carla": {
                "debug": False,
                "gpu_id": 0
            },
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Load config (no hosts.yaml file)
        settings = load_config(temp_config_dir, logger)
        
        assert settings.hosts == []

    def test_load_config_filters_comments(self, temp_config_dir, logger):
        """Test that fields starting with _ are filtered out."""
        # Create .env file
        env_content = """CARLA_PATH=/path/to/carla
CONDA_PATH=/path/to/conda
OUTPUT_PATH=/path/to/output
"""
        with open(temp_config_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create config.yaml with comment fields
        config_data = {
            "version": "2.0-dev",
            "_comment": "This is a comment",
            "filter": {
                "max_frames_per_minute": 60,
                "clustering_threshold": 0.99,
                "max_cluster_count": 30
            },
            "generator": {
                "design": "single_step_vlm",
                "steps": []
            },
            "carla": {
                "_note": "Some note",
                "debug": False,
                "gpu_id": 0
            },
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Load config
        settings = load_config(temp_config_dir, logger)
        
        # Should load successfully without comment fields
        assert settings.generator.design == "single_step_vlm"
        assert settings.carla.debug is False

    def test_load_config_version_mismatch(self, temp_config_dir, logger):
        """Test that incompatible version raises an error."""
        # Create .env file
        env_content = """CARLA_PATH=/path/to/carla
CONDA_PATH=/path/to/conda
OUTPUT_PATH=/path/to/output
"""
        with open(temp_config_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create config.yaml with incompatible version
        config_data = {
            "version": "99.0",  # Incompatible version
            "generator": {
                "design": "single_step_vlm",
                "steps": []
            },
            "carla": {
                "debug": False,
                "gpu_id": 0
            },
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Incompatible config version"):
            load_config(temp_config_dir, logger)
