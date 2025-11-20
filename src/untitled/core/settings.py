from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path


class VLMConfig(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='allow')
    
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
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')

    path: Path = Field(validation_alias="CARLA_PATH")
    username: str = Field(validation_alias="USERNAME")
    output_path: Path = Field(validation_alias="OUTPUT_PATH")


# Definition of environment variables and configuration settings
class Settings(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    # NOTE: Define settings here for type checking and auto-completion    
    vlm: VLMConfig
    carla: CARLAConfig


def load_config(root_path: Path) -> Settings:
    """
    Load configuration from a YAML file and a '.env' file.
    """
    # Load YAML configuration
    yaml_path = root_path / "config.yaml"
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    # Merge YAML data with .env environment variables
    # .env is automatically loaded by Pydantic when instantiating Settings
    settings = Settings(**yaml_data)
    return settings
