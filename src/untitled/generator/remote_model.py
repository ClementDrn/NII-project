from typing import override
from untitled.core.base import LOG
from untitled.generator.model_config import ModelConfig
from untitled.generator.model_client import ModelClient, ModelType


class RemoteModel(ModelClient):
    """
    Remote model server client (LLM/VLM server is already running elsewhere).
    """

    def __init__(self, model_type: ModelType, priority: int, batch_size: int, host: str, api_key: str, api_parameters: dict = {}):
        super().__init__(model_type, priority, batch_size, host, api_key, api_parameters)
        # We assume that remote model server is always running (started)
        self.state = ModelClient.State.STARTED

    @override
    async def start(self):
        """
        Start the remote model client (no-op for remote model).
        """
        # NOTE: Remote model is assumed to be already running and accessible
        pass

    @override
    async def stop(self):
        """
        Stop the remote model client.
        """
        # Disconnect if connected
        if self.state == ModelClient.State.CONNECTED:
            await self.close_connection()
        
        # Check if correctly disconnected
        if self.state != ModelClient.State.STARTED:
            LOG.warning("Model client status is not STARTED while stopping remote model.")

        # NOTE: We do not manage the remote model server lifecycle, therefore we do not stop it here.

    @override
    def is_parallelizable(self):
        """
        Returns True if the model client supports parallelization.
        """
        return True

    @override
    def is_remote(self) -> bool:
        """
        Returns True if the model client is remote.
        """
        return True
