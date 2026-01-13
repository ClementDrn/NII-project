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


def parse_vlm_response(chat_response: str) -> str:
    """
    Parse the VLM response to extract the relevant content after the reasoning text.
    """
    # We do not want the thinking part, before "</think>\n\n"
    # WARN: Specific to r4b model response format
    split_response = chat_response.split("</think>\n\n", 1)
    after_reasoning = split_response[-1].strip()
    return after_reasoning


class VLMClient(ABC):
    """
    Abstract base class for Vision-Language Model (VLM) clients.
    """

    class State(Enum):
        DISABLED = 0
        STARTED = 1
        CONNECTED = 2

    def __init__(self, priority: int, batch_size: int, host: str, api_key: str, api_parameters: dict = {}):
        """
        Initialize VLM client.
        
        Args:
            host: Host URL for the VLM server
            api_key: API key for authentication
            api_parameters: Additional API parameters (model, temperature, etc.)
        """
        # Attributes for the dispatcher
        self.priority = priority        # Priority for prompt dispatcher
        self.batch_size = batch_size    # Optimal number of prompts to batch together

        # Set connection parameters
        self.host = host
        self.api_key = api_key
        self.api_parameters = api_parameters
        LOG.debug(f"VLM API parameters: {self.api_parameters}")

        # Initialize OpenAI client later
        self.client = None
        self.state = VLMClient.State.DISABLED
        self.is_hot = False


    async def _send_prompt(self, system_text: str, user_text: str, user_image_paths: list[Path]) -> str:
        """
        Internal method to send a prompt to the VLM and get the response.
        
        Args:
            system_text: System message
            user_text: User message text
            user_image_paths: List of image paths to include
            
        Returns:
            Response text from VLM
        """
        # Append system text to messages
        messages = [
            {"role": "system", "content": system_text},
        ]

        # Append user images and text to messages
        user_content = []
        if user_image_paths is not None:
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
        # VLM_LOG.debug(f"Messages that will be sent to VLM: {messages}")
        
        # Send request to VLM and get response
        start_time = time.time()
        chat_response = await self.client.chat.completions.create(
            messages=messages,
            **self.api_parameters
        )
        elapsed_time = time.time() - start_time
        LOG.debug(f"VLM request completed in {elapsed_time:.2f} seconds")
        VLM_LOG.debug(f"VLM response: {chat_response}")

        # Extract and return the response text
        response_text = chat_response.choices[0].message.content
        return response_text


    @abstractmethod
    async def start(self):
        """
        Start the VLM client.
        """
        pass

    @abstractmethod
    async def stop(self):
        """
        Stop the VLM client. If connected, close the connection first.
        """
        pass

    async def set_hot(self):
        """
        Set the VLM client in a hot state.

        A hot state indicates not to free resources when stopping as the client will be reused soon.
        This is especially useful for local VLM clients to avoid starting/stopping the local server process multiple times. 
        """
        pass

    async def open_connection(self):
        """
        Open connection to VLM server. If not started, start the client first.
        """
        # Start only if disabled
        if self.state == VLMClient.State.DISABLED:
            await self.start()

        # Check if started
        if self.state != VLMClient.State.STARTED:
            LOG.warning("VLM client state is not STARTED while opening connection.")

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
            raise ConnectionError("Failed to connect to VLM server.")
        
        self.state = VLMClient.State.CONNECTED
        LOG.info("VLM connection established successfully.")

    async def close_connection(self):
        """
        Close connection to VLM server.
        """
        # Check if started
        if self.state != VLMClient.State.CONNECTED:
            LOG.warning("VLM client state is not CONNECTED while closing connection.")

        # Currently no explicit close method in AsyncOpenAI
        if self.client:
            del self.client
            self.client = None

        self.state = VLMClient.State.STARTED
        LOG.debug("VLM connection closed.")

    async def test_connection(self) -> bool:
        """
        Test connection to VLM server with a dummy prompt.
        
        Returns:
            True if connection is successful and server responds, False otherwise
        """
        # Send a simple test prompt
        try:
            LOG.info("Testing VLM connection...")
            response = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                model=self.api_parameters.get("model", "default"),
                max_tokens=10
            )
            if response and response.choices:
                LOG.info("VLM connection test successful")
                return True
            return False
        except Exception as e:
            LOG.error(f"VLM connection test failed: {e}")
            return False
        
    async def prompt(self, system_text: str, user_text: str, user_image_paths: list[Path]) -> str:
        """
        Send immediately a prompt to the VLM.
        
        Args:
            system_text: System message
            user_text: User message text
            user_image_paths: List of image paths to include
            
        Returns:
            The response text from the VLM
        """
        if self.state != VLMClient.State.CONNECTED:
            LOG.error("VLM client is not connected. Cannot send prompt.")
            raise ConnectionError("VLM client is not connected.")

        # Send the prompt and return the response
        response = await self._send_prompt(system_text, user_text, user_image_paths)
        return response

    @abstractmethod
    def is_remote(self) -> bool:
        """
        Returns True if the VLM client is remote.
        """
        pass

    def is_parallelizable(self) -> bool:
        """
        Returns True if the VLM client supports parallelization.
        """
        return False

    def get_state(self) -> State:
        """
        Get the current state of the VLM client.
        
        Returns:
            VLMClient.State enum value
        """
        return self.state