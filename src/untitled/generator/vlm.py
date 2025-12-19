"""
VLM (Vision-Language Model) module.

This module provides classes and utilities for interacting with VLM servers,
both local and remote. The main entry point is the create_vlm_client() factory function.

Module Organization:
- vlm_client.py: VLMClient abstract base class and utilities
- local_vlm.py: LocalVLM implementation for managing local vLLM servers
- remote_vlm.py: RemoteVLM implementation for connecting to remote servers
- vlm.py (this file): Re-exports for backward compatibility

Usage:
    from untitled.generator.vlm import create_vlm_client
    
    vlm = create_vlm_client(host, api_key, api_parameters)
    await vlm.start()
    await vlm.activate()
    # ... use vlm ...
    await vlm.stop()
"""

# Re-export all classes and functions for backward compatibility
from logging import config
from untitled.core.settings import ModelInstanceConfig, get_host_config_by_id
from untitled.generator.vlm_client import (
    VLMClient,
    encode_image_to_base64,
    parse_vlm_response,
)
from untitled.generator.model_config import (
    ModelConfig,
    get_model_config,
)
from untitled.generator.prompt_dispatcher import PromptRequest
from untitled.generator.local_vlm import LocalVLM
from untitled.generator.remote_vlm import RemoteVLM
from untitled.core.base import LOG, CONFIG
from untitled.generator.vlm_process import VLMProcess


class VLMClientFactory:
    """
    Factory class for creating VLMClient instances.
    This class handles underlying logic when creating new VLM clients.
    """
    _vlm_processes = {}     # Map: instance_id -> VLMProcess

    @classmethod
    def create(cls, instance_config: ModelInstanceConfig, api_parameters: dict = {}) -> VLMClient:
        """
        Factory method to create appropriate VLM client based on host.
        
        Args:
            model_name: Name of the model to use
            host: VLM server host URL
            api_key: API key for authentication
            api_parameters: Additional API parameters
        Returns:
            VLMClient instance (LocalVLM or RemoteVLM)
        """
        # Check if host is localhost
        is_local = instance_config.local_config is not None
        if not is_local and instance_config.remote_config is None:
            LOG.error("ModelInstanceConfig must have either local_config or remote_config defined.")
            return None

        # Get model config
        model_config = get_model_config(instance_config.model_name)

        # Create the VLM client depending on if it runs locally or remotely
        new_client = None
        if is_local:
            # Set host and api_key for local VLM
            host = f"http://localhost:{instance_config.local_config.port}"
            api_key = "EMPTY"
            
            # Create the local process if needed
            vlm_process = None
            if instance_config.id not in cls._vlm_processes:
                # New process
                LOG.info(f"Starting new local VLM process for instance: {instance_config.id}")
                vlm_process = VLMProcess(host, model_config, instance_config.local_config)
                cls._vlm_processes[instance_config.id] = vlm_process
            else:
                LOG.info(f"Reusing existing local VLM process for instance: {instance_config.id}")
                vlm_process = cls._vlm_processes[instance_config.id]

            # Create LocalVLM client
            new_client = LocalVLM(vlm_process, instance_config, host, api_key, api_parameters)
        else:
            # Fetch host
            host_settings = get_host_config_by_id(CONFIG, instance_config.remote_config.host_id)
            host = host_settings.host
            api_key = host_settings.api_key

            # Create RemoteVLM client
            LOG.info(f"Creating remote VLM client for host: {host_settings.host}")
            new_client = RemoteVLM(instance_config, host, api_key, api_parameters)

        return new_client


# Export create_vlm_client as a convenience function
def create_vlm_client(instance_config: ModelInstanceConfig, api_parameters: dict = {}) -> VLMClient:
    """
    Convenience function to create a VLM client.
    See VLMClientFactory.create() for details.
    """
    return VLMClientFactory.create(instance_config, api_parameters)


__all__ = [
    "VLMClient",
    "LocalVLM",
    "RemoteVLM",
    "PromptRequest",
    "encode_image_to_base64",
    "parse_vlm_response",
    "create_vlm_client",
    "VLMClientFactory",
]
