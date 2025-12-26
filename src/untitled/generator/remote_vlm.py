from typing import override
from untitled.core.base import LOG
from untitled.generator.model_config import ModelConfig
from untitled.generator.vlm_client import VLMClient


class RemoteVLM(VLMClient):
    """
    Remote VLM server client (server is already running elsewhere).
    """

    def __init__(self, priority: int, batch_size: int, host: str, api_key: str, api_parameters: dict = {}):
        super().__init__(priority, batch_size, host, api_key, api_parameters)
        # We assume that remote VLM is always running (started)
        self.state = VLMClient.State.STARTED

    @override
    async def start(self):
        """
        Start the remote VLM client (no-op for remote VLM).
        """
        # NOTE: Remote VLM is assumed to be already running and accessible
        pass

    @override
    async def stop(self):
        """
        Stop the remote VLM client.
        """
        # Disconnect if connected
        if self.state == VLMClient.State.CONNECTED:
            await self.close_connection()
        
        # Check if correctly disconnected
        if self.state != VLMClient.State.STARTED:
            LOG.warning("VLM client status is not STARTED while stopping remote VLM.")

        # NOTE: We do not manage the remote VLM server lifecycle, therefore we do not stop it here.

    @override
    def is_parallelizable(self):
        """
        Returns True if the VLM client supports parallelization.
        """
        return True

    @override
    def is_remote(self) -> bool:
        """
        Returns True if the VLM client is remote.
        """
        return True
