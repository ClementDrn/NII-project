from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path
import os
from untitled.core.version import CONFIG_VERSION, are_versions_compatible
import logging


class VLMConfig(BaseSettings):
    model_config = SettingsConfigDict(extra='allow')
    
    # Configuration parameters from config.yaml
    model: str
    temperature: float = 0.7
    max_tokens: int = 512
    extra_body: dict = {}

    # Map from .env variables
    host: str = Field(validation_alias="VLM_HOST")
    api_key: str = Field(validation_alias="VLM_API_KEY")
    
    def get_api_parameters(self) -> dict:
        """Returns only the parameters to pass to OpenAI API."""
        exclude_keys = {'host', 'api_key'}
        return {k: v for k, v in self.model_dump().items() if k not in exclude_keys}


class CARLAConfig(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore')
    
    path: Path = Field(validation_alias="CARLA_PATH")
    conda_path: str = Field(validation_alias="CONDA_PATH")
    output_path: Path = Field(validation_alias="OUTPUT_PATH")
    debug: bool = False  # Enable debug image output from SimLingo agent


class GeneratorConfig(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    # Configuration parameters for instruction generators
    design: str = "single_step_vlm"  # options: single_step_vlm, multi_step_vlm
    models: list[VLMConfig] = []


# Definition of environment variables and configuration settings
class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra='ignore')
    
    # NOTE: Define settings here for type checking and auto-completion    
    generator: GeneratorConfig
    carla: CARLAConfig


def load_config(root_path: Path, logger: logging.Logger) -> Settings:
    """
    Load configuration from a YAML file and a '.env' file.
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
    
    # Load YAML configuration
    yaml_path = root_path / "config.yaml"
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    
    # Check version compatibility
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
    
    # Create Settings instance (env vars are loaded from os.environ)
    settings = Settings(**yaml_data)
    return settings
