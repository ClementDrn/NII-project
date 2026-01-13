from untitled.core.base import VENDOR_VLM_DIR
from untitled.core.base import LOG
from untitled.generator.model_config import ModelConfig
from untitled.generator.vlm_client import VLMClient
from untitled.generator.vlm_process import VLMProcess

from typing import override
import asyncio
import os
import json
from urllib.parse import urlparse


class LocalVLM(VLMClient):
    """
    Local VLM server that can be started and stopped on the local machine.
    """

    def __init__(self, process: VLMProcess, priority: int, batch_size: int, host: str, api_key: str, api_parameters: dict = {}):
        super().__init__(priority, batch_size, host, api_key, api_parameters)
        self._process: VLMProcess = process        

    @override
    async def start(self):
        """
        Start the local VLM server.
        """
        LOG.info(f"Starting local VLM server for model {self._process.get_model_name()}...")
        
        # Start the VLM process
        await self._process.start(owner=self)
        if self._process.get_state() == VLMProcess.State.RUNNING:
            self.state = VLMClient.State.STARTED
        else:
            self.state = VLMClient.State.DISABLED
            LOG.error("Failed to start local VLM server")

    @override
    async def stop(self):
        """
        Stop the local VLM server.
        """
        # Close connection if open
        if self.state == VLMClient.State.CONNECTED:
            await self.close_connection()

        # Check if not connected anymore
        if self.state != VLMClient.State.STARTED:
            LOG.warning("VLM client status is not STARTED while stopping local VLM. Still attempting to stop...")

        # Stop the process
        # NOTE: It may not actually stop if another client is still using the same process
        await self._process.stop(owner=self)
        if self._process.get_state() == VLMProcess.State.STOPPED:
            self.state = VLMClient.State.DISABLED
            LOG.info("Local VLM server stopped")
        else:
            LOG.error("Failed to stop local VLM server")

    @override
    def is_remote(self) -> bool:
        """
        Returns True if the VLM client is remote.
        """
        return False

    @override
    def is_parallelizable(self) -> bool:
        """
        Returns True if the local VLM client supports parallel requests.
        """
        return self._process.is_parallel_to_simulator_supported()

    @override
    async def set_hot(self):
        """
        Set the VLM client in a hot state.

        A hot state indicates not to free resources that might be shared with another client when stopping as this client will start soon.
        """
        # Signal ownership to the process, so that it does not stop while in hot state
        # Typically, it means that this client may be reused soon
        await self._process.signal_ownership(owner=self)
