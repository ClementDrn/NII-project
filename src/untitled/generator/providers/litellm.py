from untitled.generator.providers.provider_adapter import ProviderAdapter
from untitled.core.base import LOG, VLM_LOG
from untitled.generator.batch import Batch, BatchRequest
from untitled.generator.model import get_model_config

from urllib.parse import urlparse
import time
from typing import Any, Optional
from litellm import acompletion, get_llm_provider
import asyncio
import random
import base64


# --- Adapter Implementation for LiteLLM/OpenAI-Compatible Providers ----------
class LiteLLMAdapter(ProviderAdapter):
    """
    Adapter implementation for LiteLLM and OpenAI-compatible providers.
    """
    _RETRY_MAX_ATTEMPTS = 10
    _RETRY_INITIAL_DELAY_SECONDS = 5.0
    _RETRY_MAX_DELAY_SECONDS = 60.0
    _RETRY_JITTER_SECONDS = 1.0

    def __init__(self, host: str, api_key: str, api_parameters: dict = {}):
        self.host = host
        self.api_key = api_key
        self.api_parameters = api_parameters
        self.client_context = None

    async def open(self) -> None:
        active_model_name = self.api_parameters.get("model", "")
        self.client_context = {
            "api_base": self._resolve_api_base(active_model_name),
            "api_key": self.api_key,
        }

    async def close(self) -> None:
        if self.client_context:
            del self.client_context
            self.client_context = None

    async def test_connection(self) -> bool:
        try:
            LOG.info("Testing model connection via LiteLLM...")
            
            max_tokens_param_name: str = (
                "max_tokens" if "max_tokens" in self.api_parameters else
                "max_completion_tokens" if "max_completion_tokens" in self.api_parameters else
                "max_output_tokens" if "max_output_tokens" in self.api_parameters else "max_tokens"
            )
            
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

    async def send_prompt(self, system_text: str = "", user_text: str = "", user_images: list[bytes] = []) -> str:
        # Construct the LiteLLM/OpenAI compatible message list
        messages: list[dict[str, Any]] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})

        user_content: list[dict[str, Any]] = []
        
        # Append images if any
        for img_bytes in user_images:
            img_base64 = base64.b64encode(img_bytes).decode("ascii")
            data_url = f"data:image/jpeg;base64,{img_base64}"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": data_url}
            })
            
        # Append text if any
        if user_text:
            user_content.append({
                "type": "text",
                "text": user_text
            })

        if user_content:
            messages.append({"role": "user", "content": user_content})

        start_time = time.time()
        request_kwargs = self._build_completion_kwargs(messages=messages)
        chat_response = await self._acompletion_with_retries(request_kwargs, timeout=1800)
        elapsed_time = time.time() - start_time

        prompt_tokens, completion_tokens, total_tokens = self._extract_token_usage(chat_response)

        LOG.debug(f"Model request completed in {elapsed_time:.2f} seconds and used {total_tokens} tokens (prompt: {prompt_tokens}, completion: {completion_tokens}).")
        VLM_LOG.debug(f"Model response: {chat_response}")
        VLM_LOG.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")

        return self._extract_response_text(chat_response)

    async def create_batch(self, batch_id: str, requests: list[BatchRequest]) -> Optional[Batch]:
        LOG.warning("Batch processing is not implemented for LiteLLM/OpenAI-compatible providers.")
        return None

    def _is_retryable_model_error(self, error: Exception) -> bool:
        error_text = f"{type(error).__name__}: {error}".lower()
        retry_indicators = (
            "overloaded", "overloaded_error", "serviceunavailable", "unavailable",
            "currently experiencing high demand", "timeout", "timed out", "connection timed out",
        )
        return any(indicator in error_text for indicator in retry_indicators)

    def _retry_wait_seconds(self, attempt_index: int) -> float:
        delay = min(
            self._RETRY_MAX_DELAY_SECONDS,
            self._RETRY_INITIAL_DELAY_SECONDS * (2 ** attempt_index),
        )
        return delay + random.uniform(0.0, self._RETRY_JITTER_SECONDS)

    async def _acompletion_with_retries(self, request_kwargs: dict, timeout: float = 1800) -> Any:
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
        request_api_parameters = dict(self.api_parameters if api_parameters is None else api_parameters)
        request_model_name = request_api_parameters.get("model", "")
        
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
        host_base = self.host.rstrip("/")
        parsed_host = urlparse(host_base)
        host_has_explicit_path = parsed_host.path not in ("", "/")
        
        if host_has_explicit_path:
            return host_base

        provider = ""
        if model_name:
            try:
                # Assuming get_llm_provider is imported/available in your full code 
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
