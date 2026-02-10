import pytest
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml
import logging

from untitled.core.settings import (
    Settings,
    GeneratorConfig,
    SimulatorConfig,
    CARLAEnvConfig,
    ModelInstanceConfig,
    LocalVLMConfig,
    RemoteVLMConfig,
    HostConfig,
    GeneratorStepConfig,
    FilterConfig,
    load_config,
    get_filter_config_by_id,
    get_generator_config_by_id,
    get_simulator_config_by_id,
    get_model_instance_config_by_id,
    get_model_instances_config_by_step_id,
    configure_settings_for_job,
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


class TestSimulatorConfig:
    """Test suite for SimulatorConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = SimulatorConfig(id="carla_basic", type="carla")
        
        assert config.id == "carla_basic"
        assert config.type == "carla"
        assert config.debug is False
        assert config.gpu_id == 0

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = SimulatorConfig(
            id="carla_debug",
            type="carla",
            debug=True,
            gpu_id=1
        )
        
        assert config.id == "carla_debug"
        assert config.type == "carla"
        assert config.debug is True
        assert config.gpu_id == 1


class TestCARLAEnvConfig:
    """Test suite for CARLAEnvConfig."""

    def test_initialization_from_env(self, monkeypatch):
        """Test that CARLAEnvConfig can be initialized from environment variables."""
        monkeypatch.setenv("CARLA_PATH", "/path/to/carla")
        monkeypatch.setenv("CONDA_PATH", "/path/to/conda")
        monkeypatch.setenv("OUTPUT_PATH", "/path/to/output")
        
        config = CARLAEnvConfig()
        
        assert config.path == Path("/path/to/carla")
        assert config.conda_path == "/path/to/conda"
        assert config.output_path == Path("/path/to/output")


class TestFilterConfig:
    """Test suite for FilterConfig."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = FilterConfig(id="default_filter")
        
        assert config.id == "default_filter"
        assert config.max_frames_per_minute == 60
        assert config.clustering_threshold == 0.990
        assert config.max_cluster_count == 30

    def test_custom_values(self):
        """Test that custom values can be set."""
        config = FilterConfig(
            id="fine_filter",
            max_frames_per_minute=240,
            clustering_threshold=0.997,
            max_cluster_count=100
        )
        
        assert config.id == "fine_filter"
        assert config.max_frames_per_minute == 240
        assert config.clustering_threshold == 0.997
        assert config.max_cluster_count == 100


class TestGeneratorConfig:
    """Test suite for GeneratorConfig."""

    def test_default_design(self):
        """Test default generator design."""
        config = GeneratorConfig(id="test_generator")
        
        assert config.id == "test_generator"
        assert config.design == "single_step_vlm"
        assert config.instruction_count_per_generation == 3
        assert config.steps == []

    def test_multi_step_design(self):
        """Test multi-step generator configuration."""
        steps = [
            GeneratorStepConfig(id="description", model_name="r4b"),
            GeneratorStepConfig(id="planning", model_name="r4b")
        ]
        config = GeneratorConfig(
            id="multi_step_gen",
            design="multi_step_vlm",
            instruction_count_per_generation=5,
            steps=steps
        )
        
        assert config.id == "multi_step_gen"
        assert config.design == "multi_step_vlm"
        assert config.instruction_count_per_generation == 5
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
        
        settings = Settings()
        
        assert settings.filters == []
        assert settings.generators == []
        assert settings.simulators == []
        assert settings.model_instances == []
        assert settings.hosts == []

    def test_full_settings(self, monkeypatch):
        """Test settings with full configuration."""
        # Set required environment variables
        monkeypatch.setenv("CARLA_PATH", "/path/to/carla")
        monkeypatch.setenv("CONDA_PATH", "/path/to/conda")
        monkeypatch.setenv("OUTPUT_PATH", "/path/to/output")
        
        # Create filter configs
        filters = [
            FilterConfig(id="fine_filter", max_frames_per_minute=240),
            FilterConfig(id="coarse_filter", max_frames_per_minute=60)
        ]
        
        # Create generator config
        steps = [
            GeneratorStepConfig(id="description", model_name="r4b"),
            GeneratorStepConfig(id="planning", model_name="r4b")
        ]
        generators = [
            GeneratorConfig(id="multi_step_gen", design="multi_step_vlm", steps=steps)
        ]
        
        # Create simulator configs
        simulators = [
            SimulatorConfig(id="carla_basic", type="carla", debug=True, gpu_id=0)
        ]
        
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
            filters=filters,
            generators=generators,
            simulators=simulators,
            model_instances=model_instances,
            hosts=hosts
        )
        
        assert len(settings.filters) == 2
        assert settings.filters[0].id == "fine_filter"
        assert len(settings.generators) == 1
        assert settings.generators[0].design == "multi_step_vlm"
        assert len(settings.generators[0].steps) == 2
        assert len(settings.simulators) == 1
        assert settings.simulators[0].debug is True
        assert len(settings.model_instances) == 2
        assert len(settings.hosts) == 1
        assert settings.model_instances[0].local_config is not None
        assert settings.model_instances[1].remote_config is not None
        assert settings.hosts[0].id == "my_vlm_host"


class TestGetConfigByIdFunctions:
    """Test suite for get_*_config_by_id functions."""

    def test_get_filter_config_by_id(self):
        """Test getting filter config by ID."""
        filters = [
            FilterConfig(id="fine_filter", max_frames_per_minute=240),
            FilterConfig(id="coarse_filter", max_frames_per_minute=60)
        ]
        settings = Settings(filters=filters)
        
        result = get_filter_config_by_id(settings, "fine_filter")
        assert result.id == "fine_filter"
        assert result.max_frames_per_minute == 240

    def test_get_filter_config_by_id_not_found(self):
        """Test getting non-existent filter config raises error."""
        settings = Settings(filters=[FilterConfig(id="existing")])
        
        with pytest.raises(ValueError, match="Filter with ID 'nonexistent' not found"):
            get_filter_config_by_id(settings, "nonexistent")

    def test_get_generator_config_by_id(self):
        """Test getting generator config by ID."""
        generators = [
            GeneratorConfig(id="gen1", design="single_step_vlm"),
            GeneratorConfig(id="gen2", design="multi_step_vlm")
        ]
        settings = Settings(generators=generators)
        
        result = get_generator_config_by_id(settings, "gen2")
        assert result.id == "gen2"
        assert result.design == "multi_step_vlm"

    def test_get_generator_config_by_id_not_found(self):
        """Test getting non-existent generator config raises error."""
        settings = Settings(generators=[GeneratorConfig(id="existing")])
        
        with pytest.raises(ValueError, match="Generator with ID 'nonexistent' not found"):
            get_generator_config_by_id(settings, "nonexistent")

    def test_get_simulator_config_by_id(self):
        """Test getting simulator config by ID."""
        simulators = [
            SimulatorConfig(id="carla_basic", type="carla"),
            SimulatorConfig(id="carla_debug", type="carla", debug=True)
        ]
        settings = Settings(simulators=simulators)
        
        result = get_simulator_config_by_id(settings, "carla_debug")
        assert result.id == "carla_debug"
        assert result.debug is True

    def test_get_simulator_config_by_id_not_found(self):
        """Test getting non-existent simulator config raises error."""
        settings = Settings(simulators=[SimulatorConfig(id="existing", type="carla")])
        
        with pytest.raises(ValueError, match="Simulator with ID 'nonexistent' not found"):
            get_simulator_config_by_id(settings, "nonexistent")

    def test_get_model_instance_config_by_id(self):
        """Test getting model instance config by ID."""
        instances = [
            ModelInstanceConfig(id="r4b_0", model_name="r4b", type="vlm", local_config=LocalVLMConfig()),
            ModelInstanceConfig(id="r4b_1", model_name="r4b", type="vlm", local_config=LocalVLMConfig())
        ]
        settings = Settings(model_instances=instances)
        
        result = get_model_instance_config_by_id(settings, "r4b_1")
        assert result.id == "r4b_1"

    def test_get_model_instance_config_by_id_not_found(self):
        """Test getting non-existent model instance config raises error."""
        settings = Settings(model_instances=[
            ModelInstanceConfig(id="existing", model_name="r4b", type="vlm", local_config=LocalVLMConfig())
        ])
        
        with pytest.raises(ValueError, match="Model instance with ID 'nonexistent' not found"):
            get_model_instance_config_by_id(settings, "nonexistent")


class TestConfigureSettingsForJob:
    """Test suite for configure_settings_for_job function."""

    def test_configure_settings_for_job_all_fields(self):
        """Test configuring settings with all job fields populated."""
        @dataclass
        class MockJob:
            simulator: str = ""
            filter: str = ""
            generator: str = ""
            model_instances: List[str] = field(default_factory=list)
        
        settings = Settings(
            simulators=[SimulatorConfig(id="carla_basic", type="carla")],
            filters=[FilterConfig(id="fine_filter")],
            generators=[GeneratorConfig(id="gen1")],
            model_instances=[
                ModelInstanceConfig(id="vlm1", model_name="model1", type="vlm", step_ids=["planning"], local_config=LocalVLMConfig()),
                ModelInstanceConfig(id="vlm2", model_name="model2", type="vlm", step_ids=["planning"], local_config=LocalVLMConfig()),
            ]
        )
        
        job = MockJob(simulator="carla_basic", filter="fine_filter", generator="gen1", model_instances=["vlm1", "vlm2"])
        configure_settings_for_job(settings, job)
        
        assert settings.simulator is not None
        assert settings.simulator.id == "carla_basic"
        assert settings.filter is not None
        assert settings.filter.id == "fine_filter"
        assert settings.generator is not None
        assert settings.generator.id == "gen1"
        assert len(settings.active_model_instances) == 2
        assert settings.active_model_instances[0].id == "vlm1"
        assert settings.active_model_instances[1].id == "vlm2"

    def test_configure_settings_for_job_optional_fields_empty(self):
        """Test configuring settings with optional fields empty."""
        @dataclass
        class MockJob:
            simulator: str = ""
            filter: str = ""
            generator: str = ""
            model_instances: List[str] = field(default_factory=list)
        
        settings = Settings(
            simulators=[SimulatorConfig(id="carla_basic", type="carla")],
            filters=[FilterConfig(id="fine_filter")],
            generators=[GeneratorConfig(id="gen1")]
        )
        
        job = MockJob(simulator="carla_basic", filter="", generator="", model_instances=[])
        configure_settings_for_job(settings, job)
        
        assert settings.simulator is not None
        assert settings.simulator.id == "carla_basic"
        assert settings.filter is None
        assert settings.generator is None
        assert len(settings.active_model_instances) == 0

    def test_configure_settings_for_job_invalid_id(self):
        """Test configuring settings with invalid ID raises error."""
        @dataclass
        class MockJob:
            simulator: str = ""
            filter: str = ""
            generator: str = ""
            model_instances: List[str] = field(default_factory=list)
        
        settings = Settings(
            simulators=[SimulatorConfig(id="carla_basic", type="carla")]
        )
        
        job = MockJob(simulator="nonexistent", filter="", generator="", model_instances=[])
        
        with pytest.raises(ValueError, match="Simulator with ID 'nonexistent' not found"):
            configure_settings_for_job(settings, job)

    def test_configure_settings_for_job_invalid_model_instance_id(self):
        """Test configuring settings with invalid model instance ID raises error."""
        @dataclass
        class MockJob:
            simulator: str = ""
            filter: str = ""
            generator: str = ""
            model_instances: List[str] = field(default_factory=list)
        
        settings = Settings(
            simulators=[SimulatorConfig(id="carla_basic", type="carla")],
            model_instances=[
                ModelInstanceConfig(id="vlm1", model_name="model1", type="vlm", local_config=LocalVLMConfig())
            ]
        )
        
        job = MockJob(simulator="carla_basic", filter="", generator="", model_instances=["nonexistent"])
        
        with pytest.raises(ValueError, match="Model instance with ID 'nonexistent' not found"):
            configure_settings_for_job(settings, job)


class TestGetModelInstancesByStepId:
    """Test suite for get_model_instances_config_by_step_id function."""

    def test_get_model_instances_by_step_id_filters_active_instances(self):
        """Test that only active model instances matching step_id are returned."""
        settings = Settings(
            model_instances=[
                ModelInstanceConfig(id="vlm1", model_name="model1", type="vlm", step_ids=["planning"], local_config=LocalVLMConfig()),
                ModelInstanceConfig(id="vlm2", model_name="model2", type="vlm", step_ids=["description"], local_config=LocalVLMConfig()),
                ModelInstanceConfig(id="vlm3", model_name="model3", type="vlm", step_ids=["planning", "description"], local_config=LocalVLMConfig()),
            ]
        )
        
        # Set active model instances (simulating what configure_settings_for_job does)
        settings.active_model_instances = [
            settings.model_instances[0],  # vlm1 - planning
            settings.model_instances[2],  # vlm3 - planning, description
        ]
        
        # Get instances for "planning" step
        planning_instances = get_model_instances_config_by_step_id(settings, "planning")
        assert len(planning_instances) == 2
        assert planning_instances[0].id == "vlm1"
        assert planning_instances[1].id == "vlm3"
        
        # Get instances for "description" step - only vlm3 is active and has description
        description_instances = get_model_instances_config_by_step_id(settings, "description")
        assert len(description_instances) == 1
        assert description_instances[0].id == "vlm3"

    def test_get_model_instances_by_step_id_empty_when_no_match(self):
        """Test that empty list is returned when no active instances match step_id."""
        settings = Settings(
            model_instances=[
                ModelInstanceConfig(id="vlm1", model_name="model1", type="vlm", step_ids=["planning"], local_config=LocalVLMConfig()),
            ]
        )
        settings.active_model_instances = [settings.model_instances[0]]
        
        # Get instances for non-existent step
        instances = get_model_instances_config_by_step_id(settings, "nonexistent_step")
        assert len(instances) == 0

    def test_get_model_instances_by_step_id_empty_when_no_active(self):
        """Test that empty list is returned when no active model instances."""
        settings = Settings(
            model_instances=[
                ModelInstanceConfig(id="vlm1", model_name="model1", type="vlm", step_ids=["planning"], local_config=LocalVLMConfig()),
            ]
        )
        # No active model instances set
        settings.active_model_instances = []
        
        instances = get_model_instances_config_by_step_id(settings, "planning")
        assert len(instances) == 0


class TestSettingsActiveProperties:
    """Test suite for Settings active property getters/setters."""

    def test_generator_property(self):
        """Test generator property getter/setter."""
        settings = Settings()
        assert settings.generator is None
        
        gen = GeneratorConfig(id="test_gen")
        settings.generator = gen
        assert settings.generator is not None
        assert settings.generator.id == "test_gen"

    def test_filter_property(self):
        """Test filter property getter/setter."""
        settings = Settings()
        assert settings.filter is None
        
        flt = FilterConfig(id="test_filter")
        settings.filter = flt
        assert settings.filter is not None
        assert settings.filter.id == "test_filter"

    def test_simulator_property(self):
        """Test simulator property getter/setter."""
        settings = Settings()
        assert settings.simulator is None
        
        sim = SimulatorConfig(id="test_sim", type="carla")
        settings.simulator = sim
        assert settings.simulator is not None
        assert settings.simulator.id == "test_sim"

    def test_active_model_instances_property(self):
        """Test active_model_instances property getter/setter."""
        settings = Settings()
        assert settings.active_model_instances == []
        
        instances = [
            ModelInstanceConfig(id="vlm1", model_name="model1", type="vlm", local_config=LocalVLMConfig()),
            ModelInstanceConfig(id="vlm2", model_name="model2", type="vlm", local_config=LocalVLMConfig()),
        ]
        settings.active_model_instances = instances
        assert len(settings.active_model_instances) == 2
        assert settings.active_model_instances[0].id == "vlm1"
        assert settings.active_model_instances[1].id == "vlm2"


class TestLoadConfig:
    """Test suite for load_config function."""

    def test_load_config_minimal(self, temp_config_dir, logger, monkeypatch):
        """Test loading minimal configuration."""
        # Clear HOSTS_PATH to ensure no hosts are loaded
        monkeypatch.delenv("HOSTS_PATH", raising=False)
        
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
            "filters": [
                {
                    "id": "default_filter",
                    "max_frames_per_minute": 60,
                    "clustering_threshold": 0.99,
                    "max_cluster_count": 30
                }
            ],
            "generators": [
                {
                    "id": "default_generator",
                    "design": "single_step_vlm",
                    "steps": []
                }
            ],
            "simulators": [
                {
                    "id": "carla_basic",
                    "type": "carla",
                    "debug": False,
                    "gpu_id": 0
                }
            ],
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Load config
        settings = load_config(temp_config_dir, logger)
        
        assert len(settings.filters) == 1
        assert settings.filters[0].id == "default_filter"
        assert len(settings.generators) == 1
        assert settings.generators[0].design == "single_step_vlm"
        assert len(settings.simulators) == 1
        assert settings.simulators[0].debug is False
        assert settings.model_instances == []
        assert settings.hosts == []

    def test_load_config_with_hosts(self, temp_config_dir, logger, monkeypatch):
        """Test loading configuration with hosts.yaml."""
        # Create .env file with HOSTS_PATH
        hosts_yaml_path = temp_config_dir / "hosts.yaml"
        env_content = f"""CARLA_PATH=/path/to/carla
CONDA_PATH=/path/to/conda
OUTPUT_PATH=/path/to/output
HOSTS_PATH={hosts_yaml_path}
"""
        with open(temp_config_dir / ".env", "w") as f:
            f.write(env_content)
        
        # Create config.yaml
        config_data = {
            "version": "2.0-dev",
            "filters": [
                {
                    "id": "fine_filter",
                    "max_frames_per_minute": 240,
                    "clustering_threshold": 0.997,
                    "max_cluster_count": 100
                }
            ],
            "generators": [
                {
                    "id": "multi_step_gen",
                    "design": "multi_step_vlm",
                    "steps": [
                        {
                            "id": "description",
                            "model_name": "r4b",
                            "temperature": 0.0,
                            "max_tokens": 1024
                        }
                    ]
                }
            ],
            "simulators": [
                {
                    "id": "carla_debug",
                    "type": "carla",
                    "debug": True,
                    "gpu_id": 0
                }
            ],
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
        
        assert len(settings.filters) == 1
        assert settings.filters[0].id == "fine_filter"
        assert len(settings.generators) == 1
        assert settings.generators[0].design == "multi_step_vlm"
        assert len(settings.generators[0].steps) == 1
        assert settings.generators[0].steps[0].id == "description"
        assert len(settings.simulators) == 1
        assert settings.simulators[0].debug is True
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
            "filters": [{"id": "default", "max_frames_per_minute": 60}],
            "generators": [{"id": "default", "design": "single_step_vlm", "steps": []}],
            "simulators": [{"id": "carla_basic", "type": "carla", "debug": False, "gpu_id": 0}],
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
            "filters": [{"id": "default", "max_frames_per_minute": 60}],
            "generators": [{"id": "default", "design": "single_step_vlm", "steps": []}],
            "simulators": [{"id": "carla_basic", "type": "carla", "debug": False, "gpu_id": 0}],
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Load config
        settings = load_config(temp_config_dir, logger)
        
        # Should load successfully without comment fields
        assert len(settings.generators) == 1
        assert settings.generators[0].design == "single_step_vlm"

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
            "filters": [],
            "generators": [],
            "simulators": [],
            "model_instances": []
        }
        with open(temp_config_dir / "config.yaml", "w") as f:
            yaml.dump(config_data, f)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Incompatible config version"):
            load_config(temp_config_dir, logger)
