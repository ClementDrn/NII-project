"""
Model (LLM/VLM) module.

This module provides classes and utilities for interacting with LLM/VLM servers,
both local and remote. The main entry point is the create_model_client() factory function.

Module Organization:
- model_client.py: ModelClient abstract base class and utilities
- local_model.py: LocalModel implementation for managing local vLLM servers
- remote_model.py: RemoteModel implementation for connecting to remote servers
- model.py (this file): Re-exports for backward compatibility

Usage:
    from untitled.generator.model import create_model_client
    
    client = create_model_client(instance_config, api_parameters)
    await client.start()
    await client.open_connection()
    # ... use client ...
    await client.stop()
"""

# Re-export all classes and functions for backward compatibility
from logging import config
from untitled.core.settings import ModelInstanceConfig, get_host_config_by_id
from untitled.generator.model_client import (
    ModelClient,
    ModelType,
    encode_image_to_base64,
    parse_model_response,
    parse_vlm_response,  # backward compatibility alias
    VLMClient,  # backward compatibility alias
)
from untitled.generator.model_config import (
    ModelConfig,
    get_model_config,
)
from untitled.generator.prompt_dispatcher import PromptRequest
from untitled.generator.local_model import LocalModel
from untitled.generator.remote_model import RemoteModel, RemoteVLM
from untitled.core.base import LOG, CONFIG
from untitled.generator.model_process import ModelProcess


class ModelClientFactory:
    """
    Factory class for creating ModelClient instances.
    This class handles underlying logic when creating new LLM/VLM clients.
    """
    _model_processes = {}     # Map: instance_id -> ModelProcess

    @classmethod
    def create(cls, instance_config: ModelInstanceConfig, api_parameters: dict = {}) -> ModelClient:
        """
        Factory method to create appropriate model client based on host.
        
        Args:
            instance_config: Configuration for the model instance
            api_parameters: Additional API parameters
        Returns:
            ModelClient instance (LocalModel or RemoteModel)
        """
        # Check if host is localhost
        is_local = instance_config.local_config is not None
        if not is_local and instance_config.remote_config is None:
            LOG.error("ModelInstanceConfig must have either local_config or remote_config defined.")
            return None

        # Get model config
        model_config: ModelConfig = get_model_config(instance_config.model_name)

        # Determine model type from instance config
        model_type = ModelType.VLM if instance_config.type == "vlm" else ModelType.LLM

        # Check that context_window does not exceed model max_model_len
        if is_local and instance_config.local_config.context_window > model_config.max_model_len:
            LOG.warning(f"Context window {instance_config.local_config.context_window} exceeds {instance_config.model_name} model's max length {model_config.max_model_len}.")

        # Get priority and batch size
        priority = instance_config.priority
        batch_size = instance_config.batch_size

        # Create the model client depending on if it runs locally or remotely
        new_client = None
        if is_local:
            # Set host and api_key for local model
            host = f"http://localhost:{instance_config.local_config.port}"
            api_key = "EMPTY"
            
            # Create the local process if needed
            model_process = None
            if instance_config.id not in cls._model_processes:
                # New process
                LOG.info(f"Starting new local model process for instance: {instance_config.id}")
                model_process = ModelProcess(host, model_config, instance_config.local_config)
                cls._model_processes[instance_config.id] = model_process
            else:
                LOG.info(f"Reusing existing local model process for instance: {instance_config.id}")
                model_process = cls._model_processes[instance_config.id]

            # Create LocalModel client
            new_client = LocalModel(model_type, model_process, priority, batch_size, host, api_key, api_parameters)
        else:
            # Fetch host
            host_settings = get_host_config_by_id(CONFIG, instance_config.remote_config.host_id)
            host = host_settings.host
            api_key = host_settings.api_key

            # Create RemoteModel client
            LOG.info(f"Creating remote model client for host: {host_settings.host}")
            new_client = RemoteModel(model_type, priority, batch_size, host, api_key, api_parameters)
        return new_client


# Export create_model_client as a convenience function
def create_model_client(instance_config: ModelInstanceConfig, api_parameters: dict = {}) -> ModelClient:
    """
    Convenience function to create a model client.
    See ModelClientFactory.create() for details.
    """
    return ModelClientFactory.create(instance_config, api_parameters)


# Alias for backward compatibility
def create_vlm_client(instance_config: ModelInstanceConfig, api_parameters: dict = {}) -> ModelClient:
    """
    Convenience function to create a VLM client.
    Deprecated: Use create_model_client() instead.
    """
    return ModelClientFactory.create(instance_config, api_parameters)


__all__ = [
    "ModelClient",
    "ModelType",
    "LocalModel",
    "RemoteModel",
    "ModelClientFactory",
    "create_model_client",
    "parse_model_response",
    # Utilities
    "PromptRequest",
    "encode_image_to_base64",
]
