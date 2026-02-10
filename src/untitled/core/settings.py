from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path
import os
from untitled.core.version import CONFIG_VERSION, HOSTS_VERSION, are_versions_compatible
import logging


class LocalVLMConfig(BaseSettings):
    """Configuration for local VLM instances."""
    model_config = ConfigDict(extra='ignore')
    
    port: int = 8000
    gpu_id: int = 0
    is_parallel_to_simulator: bool = False
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.95
    context_window: int = 8192
    max_num_seqs: int = 64
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    dtype: str = "bfloat16"


class RemoteVLMConfig(BaseSettings):
    """Configuration for remote VLM instances."""
    model_config = ConfigDict(extra='ignore')
    
    host_id: str  # References a host defined in hosts.yaml


class ModelInstanceConfig(BaseSettings):
    """Configuration for a model instance (either local or remote)."""
    model_config = ConfigDict(extra='ignore')
    
    id: str
    model_name: str
    type: str  # "vlm", "llm"
    priority: int = 0
    batch_size: int = 1
    step_ids: list[str] = []
    local_config: LocalVLMConfig | None = None
    remote_config: RemoteVLMConfig | None = None


class GeneratorStepConfig(BaseSettings):
    """Configuration for a generator step."""
    model_config = ConfigDict(extra='ignore')
    
    id: str
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 512
    thinking_mode: str = "default"  # options: default, enabled, disabled
    extra_body: dict = {}


class SimulatorConfig(BaseSettings):
    """
    Configuration for a simulator.
    
    Attributes:
        id: Unique identifier for the simulator configuration
        type: Simulator type (currently only "carla" is supported)
        debug: Enable debug image output from SimLingo agent
        gpu_id: GPU device ID to use for the simulator
    """
    model_config = ConfigDict(extra='ignore')
    
    id: str
    type: str  # Simulator type (e.g., "carla")
    debug: bool = False
    gpu_id: int = 0


class CARLAEnvConfig(BaseSettings):
    """
    Environment configuration for CARLA simulator.
    
    Contains paths and environment-specific settings loaded from .env file.
    """
    model_config = ConfigDict(extra='ignore')
    
    path: Path = Field(validation_alias="CARLA_PATH")
    conda_path: str = Field(validation_alias="CONDA_PATH")
    output_path: Path = Field(validation_alias="OUTPUT_PATH")


class FilterConfig(BaseSettings):
    """
    Configuration for frame filters.
    
    Attributes:
        id: Unique identifier for the filter configuration
        max_frames_per_minute: Number of frames per minute to retain before clustering (0 = keep all)
        clustering_threshold: Similarity threshold for clustering frames (0.0 - 1.0)
        max_cluster_count: Maximum number of clusters allowed (security measure)
    """
    model_config = ConfigDict(extra='ignore')
    
    id: str
    max_frames_per_minute: int = 60
    clustering_threshold: float = 0.990
    max_cluster_count: int = 30


class GeneratorConfig(BaseSettings):
    """
    Configuration for instruction generators.
    
    Attributes:
        id: Unique identifier for the generator configuration
        design: Generator design type ("single_step_vlm" or "multi_step_vlm")
        instruction_count_per_generation: Number of instructions to generate per generation call (>= 1)
        steps: List of generator steps configuration
    """
    model_config = ConfigDict(extra='ignore')

    id: str
    design: str = "single_step_vlm"
    instruction_count_per_generation: int = 3
    steps: list[GeneratorStepConfig] = []


class HostConfig(BaseSettings):
    """Configuration for a remote host."""
    model_config = ConfigDict(extra='ignore')
    
    id: str
    host: str
    api_key: str


class Settings(BaseSettings):
    """
    Main settings container.
    
    Contains lists of configuration objects that are referenced by ID in jobs.yaml.
    
    For backward compatibility, the following properties can be set dynamically
    to point to the active config for the current job:
    - generator: Active GeneratorConfig
    - filter: Active FilterConfig
    - simulator: Active SimulatorConfig
    """
    model_config = ConfigDict(extra='ignore')
    
    filters: list[FilterConfig] = []
    generators: list[GeneratorConfig] = []
    simulators: list[SimulatorConfig] = []
    model_instances: list[ModelInstanceConfig] = []
    hosts: list[HostConfig] = []
    carla_env: CARLAEnvConfig | None = None
    
    # Active config references (set per-job for backward compatibility)
    # TODO: Handle job context more cleanly from pipeline module and pass context to simulator/filter/generator as needed
    _active_generator: GeneratorConfig | None = None
    _active_filter: FilterConfig | None = None
    _active_simulator: SimulatorConfig | None = None
    _active_model_instances: list[ModelInstanceConfig] = []
    
    @property
    def generator(self) -> GeneratorConfig | None:
        """Get the active generator config (set per-job)."""
        return self._active_generator
    
    @generator.setter
    def generator(self, value: GeneratorConfig | None):
        """Set the active generator config."""
        self._active_generator = value
    
    @property
    def filter(self) -> FilterConfig | None:
        """Get the active filter config (set per-job)."""
        return self._active_filter
    
    @filter.setter
    def filter(self, value: FilterConfig | None):
        """Set the active filter config."""
        self._active_filter = value
    
    @property
    def simulator(self) -> SimulatorConfig | None:
        """Get the active simulator config (set per-job)."""
        return self._active_simulator
    
    @simulator.setter
    def simulator(self, value: SimulatorConfig | None):
        """Set the active simulator config."""
        self._active_simulator = value

    @property
    def active_model_instances(self) -> list[ModelInstanceConfig]:
        """Get the active model instances for the current job."""
        return self._active_model_instances
    
    @active_model_instances.setter
    def active_model_instances(self, value: list[ModelInstanceConfig]):
        """Set the active model instances."""
        self._active_model_instances = value


def load_config(root_path: Path, logger: logging.Logger) -> Settings:
    """
    Load configuration from config.yaml, hosts.yaml, and .env files.
    """
    # Load .env file into environment
    # We manually load it to avoid pydantic errors.
    env_file_path = root_path / ".env"
    if env_file_path.exists():
        with open(env_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    # Load config.yaml
    yaml_path = root_path / "config.yaml"
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    # Check version compatibility for config.yaml
    if 'version' in yaml_data:
        try:
            if not are_versions_compatible(yaml_data['version'], CONFIG_VERSION):
                logger.error(f"Config version {yaml_data['version']} is not compatible with supported version {CONFIG_VERSION}")
                raise ValueError(f"Incompatible config version: {yaml_data['version']} (supported: {CONFIG_VERSION})")
        except ValueError as e:
            logger.error(f"Version check failed: {e}")
            raise
    else:
        logger.warning("No version found in config.yaml")
    
    # Filter out comment fields (fields starting with _) and version
    yaml_data = {k: v for k, v in yaml_data.items() if not k.startswith('_') and k != 'version'}
    
    # Load hosts.yaml if it exists
    # HOSTS_PATH is in .env file
    hosts_path_str = os.environ.get("HOSTS_PATH", "")
    if hosts_path_str:
        hosts_path = Path(hosts_path_str)
        if hosts_path.exists() and hosts_path.is_file():
            with open(hosts_path, "r") as f:
                hosts_data = yaml.safe_load(f)
            
            # Check version compatibility for hosts.yaml
            if 'version' in hosts_data:
                try:
                    if not are_versions_compatible(hosts_data['version'], HOSTS_VERSION):
                        logger.error(f"Hosts version {hosts_data['version']} is not compatible with supported version {HOSTS_VERSION}")
                        raise ValueError(f"Incompatible hosts version: {hosts_data['version']} (supported: {HOSTS_VERSION})")
                except ValueError as e:
                    logger.error(f"Version check failed: {e}")
                    raise
            
            # Add hosts to yaml_data
            if 'hosts' in hosts_data:
                yaml_data['hosts'] = hosts_data['hosts']
        else:
            logger.info("No hosts.yaml file found, skipping host configuration")
            yaml_data['hosts'] = []
    else:
        logger.info("No HOSTS_PATH configured, skipping host configuration")
        yaml_data['hosts'] = []
    
    # Load CARLA environment config from environment variables
    try:
        carla_env = CARLAEnvConfig()
        yaml_data['carla_env'] = carla_env
    except Exception as e:
        logger.warning(f"Could not load CARLA environment config: {e}")
        yaml_data['carla_env'] = None
    
    # Create Settings instance
    settings = Settings(**yaml_data)
    return settings


def get_host_config_by_id(settings: Settings, host_id: str) -> HostConfig:
    """Get HostConfig by its ID."""
    for host in settings.hosts:
        if host.id == host_id:
            return host
    raise ValueError(f"Host with ID '{host_id}' not found in configuration.")


def get_filter_config_by_id(settings: Settings, filter_id: str) -> FilterConfig:
    """Get FilterConfig by its ID."""
    for filter_config in settings.filters:
        if filter_config.id == filter_id:
            return filter_config
    raise ValueError(f"Filter with ID '{filter_id}' not found in configuration.")


def get_generator_config_by_id(settings: Settings, generator_id: str) -> GeneratorConfig:
    """Get GeneratorConfig by its ID."""
    for generator in settings.generators:
        if generator.id == generator_id:
            return generator
    raise ValueError(f"Generator with ID '{generator_id}' not found in configuration.")


def get_simulator_config_by_id(settings: Settings, simulator_id: str) -> SimulatorConfig:
    """Get SimulatorConfig by its ID."""
    for simulator in settings.simulators:
        if simulator.id == simulator_id:
            return simulator
    raise ValueError(f"Simulator with ID '{simulator_id}' not found in configuration.")


def get_model_instance_config_by_id(settings: Settings, instance_id: str) -> ModelInstanceConfig:
    """Get ModelInstanceConfig by its ID."""
    for instance in settings.model_instances:
        if instance.id == instance_id:
            return instance
    raise ValueError(f"Model instance with ID '{instance_id}' not found in configuration.")


def get_model_instances_config_by_step_id(settings: Settings, step_id: str) -> list[ModelInstanceConfig]:
    """
    Get list of ModelInstanceConfig from active_model_instances that reference the given step ID.
    
    This filters the job's active model instances (set by configure_settings_for_job)
    to only those that have the specified step_id in their step_ids list.
    
    Args:
        settings: The Settings instance with active_model_instances set
        step_id: The step ID to filter by
        
    Returns:
        List of ModelInstanceConfig that are active and reference the step_id
    """
    instances = []
    for instance in settings.active_model_instances:
        if step_id in instance.step_ids:
            instances.append(instance)
    return instances


def configure_settings_for_job(settings: Settings, job) -> None:
    """
    Configure settings with active configs for a specific job.
    
    This sets the generator, filter, and simulator properties on settings
    based on the job's referenced config IDs. This provides backward
    compatibility for code that accesses CONFIG.generator, CONFIG.filter, etc.
    
    Args:
        settings: The Settings instance to configure
        job: A Job instance with simulator, filter, and generator IDs
    
    Raises:
        ValueError: If any referenced config ID is not found
    """
    # Set active simulator (required)
    if job.simulator:
        settings.simulator = get_simulator_config_by_id(settings, job.simulator)
    else:
        settings.simulator = None
    
    # Set active filter (optional)
    if job.filter:
        settings.filter = get_filter_config_by_id(settings, job.filter)
    else:
        settings.filter = None
    
    # Set active generator (optional)
    if job.generator:
        settings.generator = get_generator_config_by_id(settings, job.generator)
    else:
        settings.generator = None
    
    # Set active model instances (resolve IDs to configs)
    if job.model_instances:
        settings.active_model_instances = [
            get_model_instance_config_by_id(settings, instance_id)
            for instance_id in job.model_instances
        ]
    else:
        settings.active_model_instances = []
