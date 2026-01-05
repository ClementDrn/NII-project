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
    type: str  # "vlm"
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
    extra_body: dict = {}


class CARLAConfig(BaseSettings):
    """Configuration for CARLA simulator."""
    model_config = ConfigDict(extra='ignore')
    
    path: Path = Field(validation_alias="CARLA_PATH")
    conda_path: str = Field(validation_alias="CONDA_PATH")
    output_path: Path = Field(validation_alias="OUTPUT_PATH")
    debug: bool = False  # Enable debug image output from SimLingo agent
    gpu_id: int = 0


class FilterConfig(BaseSettings):
    """Configuration for filters."""
    model_config = ConfigDict(extra='ignore')
    
    max_frames_per_minute: int = 60
    clustering_threshold: float = 0.990
    max_cluster_count: int = 30


class GeneratorConfig(BaseSettings):
    """Configuration for instruction generators."""
    model_config = ConfigDict(extra='ignore')

    # Configuration parameters for instruction generators
    design: str = "single_step_vlm"  # options: single_step_vlm, multi_step_vlm
    steps: list[GeneratorStepConfig] = []


class HostConfig(BaseSettings):
    """Configuration for a remote host."""
    model_config = ConfigDict(extra='ignore')
    
    id: str
    host: str
    api_key: str


# Definition of environment variables and configuration settings
class Settings(BaseSettings):
    """Main settings container."""
    model_config = ConfigDict(extra='ignore')
    
    # NOTE: Define settings here for type checking and auto-completion   
    filter: FilterConfig 
    generator: GeneratorConfig
    carla: CARLAConfig
    model_instances: list[ModelInstanceConfig] = []
    hosts: list[HostConfig] = []


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
    if 'carla' in yaml_data and isinstance(yaml_data['carla'], dict):
        yaml_data['carla'] = {k: v for k, v in yaml_data['carla'].items() if not k.startswith('_')}
    
    # Load hosts.yaml if it exists
    hosts_path = root_path / "hosts.yaml"
    if hosts_path.exists():
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
    
    # Create Settings instance (env vars are loaded from os.environ)
    settings = Settings(**yaml_data)
    return settings

def get_host_config_by_id(settings: Settings, host_id: str) -> HostConfig:
    """
    Get HostConfig by its ID.
    """
    for host in settings.hosts:
        if host.id == host_id:
            return host
    raise ValueError(f"Host with ID '{host_id}' not found in configuration.")

def get_model_instances_config_by_step_id(settings: Settings, step_id: str) -> list[ModelInstanceConfig]:
    """
    Get list of ModelInstanceConfig that reference the given step ID.
    """
    instances = []
    for instance in settings.model_instances:
        if step_id in instance.step_ids:
            instances.append(instance)
    return instances
