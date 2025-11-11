from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path


class VLMConfig(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    # Configuration parameters from config.yaml
    model: str
    temperature: float = 0.7
    max_tokens: int = 512
    enable_thinking: str = "auto"      # "true", "false", or "auto"

    # Map from .env variables
    host: str = Field(validation_alias="VLM_HOST")
    api_key: str = Field(validation_alias="VLM_API_KEY")


# Definition of environment variables and configuration settings
class Settings(BaseSettings):
    # Load .env file
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    
    # NOTE: Define settings here for type checking and auto-completion    
    vlm: VLMConfig


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
