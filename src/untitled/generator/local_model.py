from untitled.core.base import VENDOR_VLM_DIR
from untitled.core.base import LOG
from untitled.generator.model_config import ModelConfig
from untitled.generator.model_client import ModelClient, ModelType
from untitled.generator.model_process import ModelProcess
from untitled.generator.providers.provider_adapter import ProviderAdapter

from typing import override
import asyncio
import os
import json
from urllib.parse import urlparse


class LocalModel(ModelClient):
    """
    Local model server (LLM/VLM) that can be started and stopped on the local machine.
    """

    def __init__(self, model_type: ModelType, process: ModelProcess, priority: int, batch_size: int, adapter: ProviderAdapter):
        super().__init__(model_type, priority, batch_size, adapter)
        self._process: ModelProcess = process        

    @override
    async def start(self):
        """
        Start the local model server.
        """
        LOG.info(f"Starting local model server for model {self._process.get_model_name()}...")
        
        # Start the model process
        await self._process.start(owner=self)
        if self._process.get_state() == ModelProcess.State.RUNNING:
            self.state = ModelClient.State.STARTED
        else:
            self.state = ModelClient.State.DISABLED
            LOG.error("Failed to start local model server")

    @override
    async def stop(self):
        """
        Stop the local model server.
        """
        # Close connection if open
        if self.state == ModelClient.State.CONNECTED:
            await self.close_connection()

        # Check if not connected anymore
        if self.state != ModelClient.State.STARTED:
            LOG.warning("Model client status is not STARTED while stopping local model. Still attempting to stop...")

        # Stop the process
        # NOTE: It may not actually stop if another client is still using the same process
        await self._process.stop(owner=self)
        if self._process.get_state() == ModelProcess.State.STOPPED:
            self.state = ModelClient.State.DISABLED
            LOG.info("Local model server stopped")
        else:
            LOG.error("Failed to stop local model server")

    @override
    def is_remote(self) -> bool:
        """
        Returns True if the model client is remote.
        """
        return False

    @override
    def is_parallelizable(self) -> bool:
        """
        Returns True if the local model client supports parallel requests.
        """
        return self._process.is_parallel_to_simulator_supported()

    @override
    async def set_hot(self):
        """
        Set the model client in a hot state.

        A hot state indicates not to free resources that might be shared with another client when stopping as this client will start soon.
        """
        # Signal ownership to the process, so that it does not stop while in hot state
        # Typically, it means that this client may be reused soon
        await self._process.signal_ownership(owner=self)
