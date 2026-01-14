from untitled.core.base import LOG, VLM_LOG
from untitled.generator.model_config import ModelConfig, get_model_config

from abc import ABC, abstractmethod
import base64
from PIL import Image
from openai import AsyncOpenAI
import io
import time
import asyncio
import os
from pathlib import Path
from enum import Enum


class ModelType(Enum):
    """Type of model: VLM (Vision-Language Model) or LLM (Language Model)."""
    VLM = "vlm"
    LLM = "llm"


def encode_image_to_base64(image_path: Path) -> str:
    """
    Encode an image to a base64 string.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode("ascii")
        data_url = f"data:image/jpeg;base64,{img_base64}"
        return data_url


def parse_model_response(chat_response: str) -> str:
    """
    Parse the model response to extract the relevant content after the reasoning text.
    """
    # We do not want the thinking part, before "</think>\n\n"
    # WARN: Specific to r4b model response format
    split_response = chat_response.split("</think>\n\n", 1)
    after_reasoning = split_response[-1].strip()
    return after_reasoning


# Alias for backward compatibility
parse_vlm_response = parse_model_response


class ModelClient(ABC):
    """
    Abstract base class for LLM/VLM (Language/Vision-Language Model) clients.
    """

    class State(Enum):
        DISABLED = 0
        STARTED = 1
        CONNECTED = 2

    def __init__(self, model_type: ModelType, priority: int, batch_size: int, host: str, api_key: str, api_parameters: dict = {}):
        """
        Initialize model client.
        
        Args:
            model_type: Type of model (VLM or LLM)
            host: Host URL for the model server
            api_key: API key for authentication
            api_parameters: Additional API parameters (model, temperature, etc.)
        """
        # Model type
        self.model_type = model_type

        # Attributes for the dispatcher
        self.priority = priority        # Priority for prompt dispatcher
        self.batch_size = batch_size    # Optimal number of prompts to batch together

        # Set connection parameters
        self.host = host
        self.api_key = api_key
        self.api_parameters = api_parameters
        LOG.debug(f"Model API parameters: {self.api_parameters}")

        # Initialize OpenAI client later
        self.client = None
        self.state = ModelClient.State.DISABLED
        self.is_hot = False


    async def _send_prompt(self, system_text: str, user_text: str, user_image_paths: list[Path]) -> str:
        """
        Internal method to send a prompt to the model and get the response.
        
        Args:
            system_text: System message
            user_text: User message text
            user_image_paths: List of image paths to include (only for VLM)
            
        Returns:
            Response text from model
        """
        # Append system text to messages
        messages = [
            {"role": "system", "content": system_text},
        ]

        # Append user images and text to messages
        user_content = []
        if user_image_paths is not None:
            # Check if model supports images
            if self.model_type == ModelType.LLM:
                LOG.warning("Images provided but model type is LLM. Images will be ignored.")
            else:
                for image_path in user_image_paths:
                    data_url = encode_image_to_base64(image_path)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        },
                    })
        if user_text is not None:
            user_content.append({
                "type": "text",
                "text": user_text,
            })
        messages.append({
            "role": "user",
            "content": user_content,
        })
        # VLM_LOG.debug(f"Messages that will be sent to model: {messages}")
        
        # Send request to model and get response
        start_time = time.time()
        chat_response = await self.client.chat.completions.create(
            messages=messages,
            **self.api_parameters
        )
        elapsed_time = time.time() - start_time
        LOG.debug(f"Model request completed in {elapsed_time:.2f} seconds")
        VLM_LOG.debug(f"Model response: {chat_response}")

        # Extract and return the response text
        response_text = chat_response.choices[0].message.content
        return response_text


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

        # Set OpenAI's API key and API base to use vLLM's API server
        openai_api_base = self.host + "/v1"
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=openai_api_base,
        )

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

        # Currently no explicit close method in AsyncOpenAI
        if self.client:
            del self.client
            self.client = None

        self.state = ModelClient.State.STARTED
        LOG.debug("Model connection closed.")

    async def test_connection(self) -> bool:
        """
        Test connection to model server with a dummy prompt.
        
        Returns:
            True if connection is successful and server responds, False otherwise
        """
        # Send a simple test prompt
        try:
            LOG.info("Testing model connection...")
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                model=self.api_parameters.get("model", "default"),
                max_tokens=10
            )
            if response and response.choices:
                LOG.info("Model connection test successful")
                return True
            return False
        except Exception as e:
            LOG.error(f"Model connection test failed: {e}")
            return False
        
    async def prompt(self, system_text: str, user_text: str, user_image_paths: list[Path]) -> str:
        """
        Send immediately a prompt to the model.
        
        Args:
            system_text: System message
            user_text: User message text
            user_image_paths: List of image paths to include (only for VLM, ignored for LLM)
            
        Returns:
            The response text from the model
        """
        if self.state != ModelClient.State.CONNECTED:
            LOG.error("Model client is not connected. Cannot send prompt.")
            raise ConnectionError("Model client is not connected.")

        # Send the prompt and return the response
        response = await self._send_prompt(system_text, user_text, user_image_paths)
        return response

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
