from untitled.core.base import LOG, VLM_LOG
from untitled.generator.model_config import get_model_config

from abc import ABC, abstractmethod
from typing import Any, Optional
import base64
from PIL import Image
from litellm import acompletion, get_llm_provider
import io
import time
import random
import asyncio
from pathlib import Path
from enum import Enum
from urllib.parse import urlparse


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

    _RETRY_MAX_ATTEMPTS = 10    # up to 1.5 hours of retries
    _RETRY_INITIAL_DELAY_SECONDS = 5.0
    _RETRY_MAX_DELAY_SECONDS = 60.0
    _RETRY_JITTER_SECONDS = 1.0

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
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_text},
        ]

        # Append user images and text to messages
        user_content: list[dict[str, Any]] = []
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
        # LOG.debug(f"Sending prompt to model at {self.host} with API parameters: {self.api_parameters}...")
        
        # Send request to model and get response
        start_time = time.time()
        request_kwargs = self._build_completion_kwargs(messages=messages)
        chat_response = await self._acompletion_with_retries(request_kwargs, timeout=1800)  # timeout = 30 minutes
        elapsed_time = time.time() - start_time

        # Collect token count data
        prompt_tokens, completion_tokens, total_tokens = self._extract_token_usage(chat_response)

        # Log tokens and time
        LOG.debug(f"Model request completed in {elapsed_time:.2f} seconds and used {total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens}).")
        VLM_LOG.debug(f"Model response: {chat_response}")
        VLM_LOG.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

        # Extract and return the response text
        response_text = self._extract_response_text(chat_response)
        return response_text

    def _is_retryable_model_error(self, error: Exception) -> bool:
        """
        Return True for transient model/provider failures that should be retried.

        Handles common overload and timeout variants returned by LiteLLM providers
        (e.g. Gemini UNAVAILABLE spikes, Anthropic overloaded/timeout errors).
        """
        error_text = f"{type(error).__name__}: {error}".lower()
        retry_indicators = (
            "overloaded",
            "overloaded_error",
            "serviceunavailable",
            "unavailable",
            "currently experiencing high demand",
            "timeout",
            "timed out",
            "connection timed out",
        )
        return any(indicator in error_text for indicator in retry_indicators)

    def _retry_wait_seconds(self, attempt_index: int) -> float:
        """
        Exponential backoff with bounded jitter.
        """
        delay = min(
            self._RETRY_MAX_DELAY_SECONDS,
            self._RETRY_INITIAL_DELAY_SECONDS * (2 ** attempt_index),
        )
        return delay + random.uniform(0.0, self._RETRY_JITTER_SECONDS)

    async def _acompletion_with_retries(self, request_kwargs: dict, timeout: float = 1800) -> Any:
        """
        Execute LiteLLM completion with retries on transient overload/timeout errors.
        """
        for attempt_index in range(self._RETRY_MAX_ATTEMPTS + 1):
            try:
                return await acompletion(**request_kwargs, timeout=timeout)
            except Exception as error:
                is_retryable = self._is_retryable_model_error(error)
                has_more_attempts = attempt_index < self._RETRY_MAX_ATTEMPTS
                if not is_retryable or not has_more_attempts:
                    raise

                wait_seconds = self._retry_wait_seconds(attempt_index)
                LOG.warning(
                    "Transient model error on attempt %d/%d (%s). Waiting %.1fs before retry.",
                    attempt_index + 1,
                    self._RETRY_MAX_ATTEMPTS + 1,
                    error,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

    def _build_completion_kwargs(self, messages: list[dict], api_parameters: dict | None = None) -> dict:
        """
        Build LiteLLM completion kwargs with provider-independent API settings.
        """
        request_api_parameters = dict(self.api_parameters if api_parameters is None else api_parameters)
        request_model_name = request_api_parameters.get("model", "")
        # Temporary security check 
        if isinstance(request_model_name, str) and "/" not in request_model_name:
            try:
                request_api_parameters["model"] = get_model_config(request_model_name).model_name
            except ValueError:
                pass

        request_model_name = request_api_parameters.get("model", "")
        request_kwargs = {
            "messages": messages,
            **request_api_parameters,
        }
        resolved_api_base = self._resolve_api_base(request_model_name)
        if resolved_api_base is not None:
            request_kwargs.setdefault("api_base", resolved_api_base)
        request_kwargs.setdefault("api_key", self.api_key)
        return request_kwargs

    def _resolve_api_base(self, model_name: str) -> Optional[str]:
        """
        Resolve API base URL from host and model provider for LiteLLM.

        - If host already has a path (e.g. reverse proxy), preserve it as-is.
        - Otherwise, use LiteLLM provider inference from `model_name`.
        - OpenAI-compatible providers default to `/v1`.
        - Provider-native cloud APIs (Gemini/Anthropic) can use LiteLLM defaults.
        - Other providers use host root.
        """
        host_base = self.host.rstrip("/")

        parsed_host = urlparse(host_base)
        host_has_explicit_path = parsed_host.path not in ("", "/")
        if host_has_explicit_path:
            return host_base

        provider = ""
        if model_name:
            try:
                _, provider, _, _ = get_llm_provider(
                    model=model_name,
                    api_base=host_base,
                    api_key=self.api_key,
                )
            except Exception:
                provider = model_name.split("/", 1)[0].lower() if "/" in model_name else ""

        openai_compatible_providers = {"", "openai", "text-completion-openai", "azure", "azure_ai"}
        if provider in openai_compatible_providers:
            return f"{host_base}/v1"

        provider_default_base = {"gemini", "anthropic"}
        if provider in provider_default_base:
            return None

        return host_base

    def _extract_token_usage(self, chat_response: Any) -> tuple[int, int, int]:
        """
        Extract token usage from LiteLLM/OpenAI-like responses.
        """
        usage = getattr(chat_response, "usage", None)
        if usage is None and isinstance(chat_response, dict):
            usage = chat_response.get("usage")

        if usage is None:
            return -1, -1, -1

        if isinstance(usage, dict):
            return (
                usage.get("prompt_tokens", -1),
                usage.get("completion_tokens", -1),
                usage.get("total_tokens", -1),
            )

        return (
            getattr(usage, "prompt_tokens", -1),
            getattr(usage, "completion_tokens", -1),
            getattr(usage, "total_tokens", -1),
        )

    def _extract_response_text(self, chat_response: Any) -> str:
        """
        Extract response text from LiteLLM/OpenAI-like responses.
        """
        choices = getattr(chat_response, "choices", None)
        if choices is None and isinstance(chat_response, dict):
            choices = chat_response.get("choices")

        if not choices:
            return ""

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None and isinstance(first_choice, dict):
            message = first_choice.get("message", {})

        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")

        if isinstance(content, list):
            text_parts = []
            for content_item in content:
                if isinstance(content_item, dict) and content_item.get("type") == "text":
                    text_parts.append(content_item.get("text", ""))
            return "\n".join(part for part in text_parts if part)

        return content or ""


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

        # Store active connection context for LiteLLM request execution
        active_model_name = self.api_parameters.get("model", "")
        self.client = {
            "api_base": self._resolve_api_base(active_model_name),
            "api_key": self.api_key,
        }

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

        # LiteLLM uses stateless request calls, so there is no persistent transport to close
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

            # Use provider-specific output-token parameter when available
            max_tokens_param_name: str = (
                "max_tokens" if "max_tokens" in self.api_parameters else
                "max_completion_tokens" if "max_completion_tokens" in self.api_parameters else
                "max_output_tokens" if "max_output_tokens" in self.api_parameters else ""
            )
            if max_tokens_param_name == "":
                LOG.warning("No max tokens parameter found in API parameters. Defaulting to 'max_tokens'.")
                max_tokens_param_name = "max_tokens"

            LOG.debug(f"Using max tokens parameter name: {max_tokens_param_name}")
            connection_test_params = dict(self.api_parameters)
            connection_test_params.pop("max_tokens", None)
            connection_test_params.pop("max_completion_tokens", None)
            connection_test_params.pop("max_output_tokens", None)
            connection_test_params[max_tokens_param_name] = 5
            connection_test_params.setdefault("model", self.api_parameters.get("model", "default"))

            response = await self._acompletion_with_retries(
                self._build_completion_kwargs(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello"}
                    ],
                    api_parameters=connection_test_params,
                ),
                timeout=60,
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

        # Send the prompt and wait for the response
        response = await self._send_prompt(system_text, user_text, user_image_paths)
        if response is None:
            LOG.warning("Received empty response from model.")

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


# Backward compatibility alias
# WARN: Deprecated TODO: Remove for release
VLMClient = ModelClient
