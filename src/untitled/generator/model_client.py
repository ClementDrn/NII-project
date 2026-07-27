from untitled.core.base import LOG, VLM_LOG
from untitled.generator.batch import Batch, BatchRequest
from untitled.generator.providers.provider_adapter import ProviderAdapter
from untitled.generator.request import Request

from abc import ABC, abstractmethod
import io
from PIL import Image
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    """Type of model: VLM (Vision-Language Model) or LLM (Language Model)."""
    VLM = "vlm"
    LLM = "llm"


def parse_model_response(chat_response: str) -> str:
    """
    Parse the model response to extract the relevant content after the reasoning text.
    """
    # We do not want the thinking part, before "</think>\n\n"
    # WARN: Specific to qwen3 model response format
    split_response = chat_response.split("</think>\n\n", 1)
    after_reasoning = split_response[-1].strip()
    return after_reasoning

# Alias for backward compatibility
# WARN: Deprecated TODO: Remove for release
parse_vlm_response = parse_model_response


class ModelClient(ABC):
    """
    Abstract base class for LLM/VLM (Language/Vision-Language Model) clients.
    """

    class State(Enum):
        DISABLED = 0
        STARTED = 1
        CONNECTED = 2

    def __init__(self, model_type: ModelType, priority: int, batch_size: int, adapter: ProviderAdapter):
        """
        Initialize model client.
        
        Args:
            model_type: Type of model (VLM or LLM)
            priority: Priority for prompt dispatcher
            batch_size: Optimal number of prompts to batch together
            adapter: Provider configuration handler
        """
        # Model type
        self.model_type = model_type

        # Attributes for the dispatcher
        self.priority = priority        
        self.batch_size = batch_size    
        
        # Provider Adapter strategy
        self.adapter = adapter
        
        self.state = ModelClient.State.DISABLED
        self.is_hot = False

    @abstractmethod
    async def start(self):
        """
        Start the model client.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Stop the model client. If connected, close the connection first.
        """
        pass

    async def set_hot(self):
        """
        Set the model client in a hot state.

        A hot state indicates not to free resources when stopping as the client will be reused soon.
        This is especially useful for local model clients to avoid starting/stopping the local server process multiple times. 
        """
        pass

    async def open_connection(self):
        """
        Open connection to model server. If not started, start the client first.
        """
        # Start only if disabled
        if self.state == ModelClient.State.DISABLED:
            await self.start()

        # Check if started
        if self.state != ModelClient.State.STARTED:
            LOG.warning("Model client state is not STARTED while opening connection.")

        # Let the adapter execute any necessary opening handshakes
        await self.adapter.open()

        # Test connection
        success = await self.test_connection()
        if not success:
            # Raise connection error
            # NOTE: open_connection needs to be awaited in order to catch this exception.
            raise ConnectionError("Failed to connect to model server.")
        
        self.state = ModelClient.State.CONNECTED
        LOG.info("Model connection established successfully.")

    async def close_connection(self):
        """
        Close connection to model server.
        """
        # Check if started
        if self.state != ModelClient.State.CONNECTED:
            LOG.warning("Model client state is not CONNECTED while closing connection.")

        await self.adapter.close()

        self.state = ModelClient.State.STARTED
        LOG.debug("Model connection closed.")

    async def test_connection(self) -> bool:
        """
        Test connection to model server with a dummy prompt.
        
        Returns:
            True if connection is successful and server responds, False otherwise
        """
        return await self.adapter.test_connection()
        
    async def prompt(self, request: Request) -> str:
        """
        Send immediately a prompt to the model.
        
        Args:
            request: The request object containing system and user content
            
        Returns:
            The response text from the model
        """
        if self.state != ModelClient.State.CONNECTED:
            LOG.error("Model client is not connected. Cannot send prompt.")
            raise ConnectionError("Model client is not connected.")

        # Convert image paths to formatted base64 urls
        if self.model_type == ModelType.LLM:
            for block in request.user_content:
                if block.type == "image":
                    LOG.warning(f"LLM model client received an image block. The following image will be ignored: {block.image_path}")

        # Delegate request entirely to the provider strategy
        response = await self.adapter.send_prompt(request)
        return response

    async def batch(self, requests: list[BatchRequest]) -> Batch:
        """
        Create a batch for processing multiple requests.
        
        Args:
            batch_id: Unique identifier for the batch
            requests: List of BatchRequest objects

        Returns:
            A Batch object representing the created batch
        """
        if self.state != ModelClient.State.CONNECTED:
            LOG.error("Model client is not connected. Cannot create batch.")
            raise ConnectionError("Model client is not connected.")

        batch_object = await self.adapter.create_batch(requests)
        if batch_object is None:
            LOG.error("Failed to create batch object.")
            raise RuntimeError("Failed to create batch object.")
        
        return batch_object

    @abstractmethod
    def is_remote(self) -> bool:
        """
        Returns True if the model client is remote.
        """
        pass

    def is_parallelizable(self) -> bool:
        """
        Returns True if the model client supports parallelization.
        """
        return False

    def get_state(self) -> State:
        """
        Get the current state of the model client.
        
        Returns:
            ModelClient.State enum value
        """
        return self.state


# Backward compatibility alias
# WARN: Deprecated TODO: Remove for release
VLMClient = ModelClient
