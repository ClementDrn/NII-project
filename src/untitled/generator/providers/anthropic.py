from untitled.generator.providers.provider_adapter import ProviderAdapter
from untitled.generator.batch import Batch, BatchRequest, BatchRequestStatus, BatchStatus, BatchRequestResult
from untitled.core.base import LOG, VLM_LOG

import anthropic
from anthropic.types.messages.batch_create_params import Request as AnthropicRequest
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming as AnthropicMessageCreateParamsNonStreaming
from anthropic.types.message_create_params import MessageCreateParamsStreaming as AnthropicMessageCreateParamsStreaming
from anthropic.types.messages.message_batch import MessageBatch as AnthropicMessageBatch
from anthropic.types.content_block import ContentBlock as AnthropicContentBlock

import time
from typing import Any, AsyncIterator, List
import base64


# --- Utils -------------------------------------------------------------------

def _convert_user_messages_to_anthropic_format(user_text: str, user_images: list[bytes]) -> list[dict[str, Any]]:
    """
    Translates user messages to Anthropic format.

    Args:
        user_text: The user prompt text
        user_images: List of user-provided images in bytes format

    Returns:
        Tuple of anthropic_messages_array.
    """
    anthropic_messages = []

    # Process user text
    if user_text:
        anthropic_messages.append({"role": "user", "content": user_text})

    # Process user images
    for image_bytes in user_images:
        # Convert image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        anthropic_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",     # WARN: Assuming JPEG
                        "data": base64_image
                    }
                }
            ]
        })

    return anthropic_messages

def _clean_model_name(model_name: str) -> str:
    """
    Clean the model name by removing any provider-specific prefixes.

    Args:
        model_name: The raw model name

    Returns:
        Cleaned model name without provider prefixes.
    """
    if model_name.startswith("anthropic/"):
        return model_name[len("anthropic/"):]
    return model_name

def _build_anthropic_messages_kwargs(system_text: str, user_text: str, user_images: list[bytes], api_parameters: dict) -> dict:
    """
    Build the kwargs for the Anthropic messages.create() call.

    Args:
        system_text: The system prompt text
        user_text: The user prompt text
        user_images: List of user-provided images in bytes format
        api_parameters: Optional dictionary of additional API parameters.

    Returns:
        A dictionary of parameters suitable for the Anthropic messages.create() call.
    """
    # Anthropic requires model and max_tokens
    model = _clean_model_name(api_parameters.get("model", ""))
    if not model:
        raise ValueError("Model parameter is required for Anthropic API calls.")

    max_tokens = api_parameters.get("max_tokens", 4096)
    kwargs = {
        "model": model,
        "max_tokens": max_tokens
    }
    
    # Include system and user messages
    anthropic_messages = _convert_user_messages_to_anthropic_format(user_text, user_images)
    kwargs["messages"] = anthropic_messages
    if system_text:
        kwargs["system"] = system_text

    # Include other supported API parameters
    # NOTE: Add additional parameters here if needed
    allowed_keys = {"temperature", "top_p", "top_k", "user_profile_id", "thinking", "timeout"}
    for key, value in api_parameters.items():
        if key in allowed_keys and key not in kwargs:
            kwargs[key] = value
    
    return kwargs

def _extract_anthropic_response_content(content: List[AnthropicContentBlock]) -> tuple[str, str]:
    """
    Extracts text and thinking content from a list of Anthropic content blocks.

    Returns:
        Tuple of (text_content, thinking_content) extracted from the content blocks.
    """
    text_content = ""
    thinking_content = ""

    for block in content:
        match block.type:
            case "text":
                text_content += block.text
            case "thinking":
                thinking_content += block.thinking

    return text_content, thinking_content


# --- Anthropic Batch ---------------------------------------------------------

class AnthropicBatch(Batch):
    """Implementation of the Batch class for Anthropic model clients."""
    
    def __init__(self, client: anthropic.AsyncAnthropic, api_parameters: dict, requests: list[BatchRequest]):
        super().__init__(requests)

        # Save Anthropic SDK client
        self._client: anthropic.AsyncAnthropic = client
        if not self._client.api_key:
            raise ValueError("AnthropicBatch requires a client that is initialized with an API key.")

        # Fetch max_tokens and model from api_parameters
        self._max_tokens = api_parameters.get("max_tokens", 4096)
        self._model = _clean_model_name(api_parameters.get("model", ""))
        if not self._model:
            raise ValueError("Model parameter is required for Anthropic batch processing.")

    @staticmethod
    def _map_anthropic_status_to_batch_status(anthropic_status: str) -> BatchStatus:
        """Map Anthropic's processing status to the normalized BatchStatus."""
        mapping = {
            "in_progress": BatchStatus.RUNNING,
            "canceling": BatchStatus.CANCELLED,
            "ended": BatchStatus.COMPLETED,
        }
        return mapping.get(anthropic_status, BatchStatus.UNKNOWN)

    async def send(self) -> BatchStatus:
        """Send the batch to the model client."""
        # Prepare the requests into the Anthropic batch format
        anthropic_requests = []
        for request in self._requests:
            # Prepare parameters for currentrequest
            kwargs = _build_anthropic_messages_kwargs(
                system_text=request.system_text,
                user_text=request.user_text or "",
                user_images=request.user_images or [],
                api_parameters={"model": self._model, "max_tokens": self._max_tokens}   # WARN: Only model and max_tokens are passed as api_parameters for batches
            )
            # Add request to the batch
            anthropic_requests.append(
                AnthropicRequest(
                    custom_id=request.request_id,
                    params=AnthropicMessageCreateParamsNonStreaming(**kwargs),
                )
            )

        # Create and send the batch
        message_batch: AnthropicMessageBatch = await self._client.messages.batches.create(
            requests=anthropic_requests
        )

        # Save id
        self._batch_id = message_batch.id

        # Return status
        return self._map_anthropic_status_to_batch_status(message_batch.processing_status)

    async def check_status(self) -> BatchStatus:
        """Check the status of the batch."""
        message_batch: AnthropicMessageBatch = await self._client.messages.batches.retrieve(self._batch_id)
        return self._map_anthropic_status_to_batch_status(message_batch.processing_status)

    # NOTE: despite the "async def" keywords, there is no need to await 
    async def download_results(self) -> AsyncIterator[BatchRequestResult]:
        """
        Download the results of the batch if completed.
        If the batch is not completed, behavior is undefined.
        
        Returns:
            An iterator over the results, yielding each result's content as a list of dictionaries (JSONL format) upon success.
            Content for batch messages that failed is an empty list.
        """
        # Stream results file in memory-efficient chunks, processing one at a time
        async for result in await self._client.messages.batches.results(self._batch_id):
            # e.g. {"custom_id":"my-second-request","result":{"type":"succeeded","message":{"id":"msg_014VwiXbi91y3JMjcpyGBHX5","type":"message","role":"assistant","model":"claude-opus-4-8","content":[{"type":"text","text":"Hello again! It's nice to see you. How can I assist you today? Is there anything specific you'd like to chat about or any questions you have?"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":36}}}}
            return_status: BatchRequestStatus = BatchRequestStatus.UNKNOWN
            return_text: str = ""
            return_thinking: str = ""
            
            match result.result.type:
                case "succeeded":
                    LOG.debug(f"Success! Downloaded batch results for '{result.custom_id}'...")
                    return_status = BatchRequestStatus.SUCCEEDED
                    # Save content
                    return_text, return_thinking = _extract_anthropic_response_content(result.result.message.content)  # type: ignore

                case "errored":
                    if result.result.error.error.type == "invalid_request_error":   # type: ignore
                        # NOTE: Request body must be fixed before re-sending request
                        LOG.error(f"Validation error {result.custom_id}")
                    else:
                        # NOTE: Request can be retried directly
                        LOG.error(f"Server error {result.custom_id}")
                    return_status = BatchRequestStatus.ERRORED
                case "expired":
                    LOG.error(f"Request expired {result.custom_id}")
                    return_status = BatchRequestStatus.EXPIRED
                case "cancelled":
                    LOG.error(f"Request cancelled {result.custom_id}")
                    return_status = BatchRequestStatus.CANCELLED
                case _:
                    LOG.error(f"Unknown result type '{result.result.type}' for request '{result.custom_id}'")

            yield BatchRequestResult(
                request_id=result.custom_id,
                status=return_status,
                text=return_text,
                thinking=return_thinking
            )


# --- Anthropic Model Adapter -------------------------------------------------

class AnthropicAdapter(ProviderAdapter):
    """
    Adapter implementation for the official Anthropic API.
    Converts OpenAI-formatted message arrays into Anthropic's native schema.
    """
    
    def __init__(self, api_key: str, api_parameters: dict = {}):
        self.api_key: str = api_key
        self.api_parameters: dict = api_parameters
        self.client: anthropic.AsyncAnthropic | None = None

        # Fetch model from api_parameters and check it is present
        self._model = _clean_model_name(api_parameters.get("model", ""))
        if not self._model:
            raise ValueError("Model parameter is required for AnthropicAdapter initialization.")


    async def open(self) -> None:
        """Instantiate the async Anthropic client."""
        self.client = anthropic.AsyncAnthropic(api_key=self.api_key)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self.client:
            await self.client.close()
            self.client = None

    async def test_connection(self) -> bool:
        """Execute a minimal prompt to verify provider connectivity."""
        if not self.client:
            LOG.error("Client not initialized. Call open() first.")
            return False

        try:            
            LOG.info(f"Testing model connection of {self._model} via Anthropic SDK...")            

            # Anthropic requires max_tokens
            max_tokens = self.api_parameters.get("max_tokens", 10)

            response = await self.client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": "Hello"}],
                timeout=30.0
            )
            
            if response and response.content:
                LOG.info("Model connection test successful")
                return True
            return False
        except Exception as e:
            LOG.error(f"Anthropic connection test failed: {e}")
            return False

    async def send_prompt(self, system_text: str = "", user_text: str = "", user_images: list[bytes] = []) -> str:
        """
        Send a message payload to the Anthropic API and return the text response.
        
        Args:
            system_text: The system prompt text
            user_text: The user prompt text
            user_images: List of user-provided images in bytes format

        Returns:
            The extracted text response from the model.
        """
        if not self.client:
            raise ConnectionError("Anthropic client is not initialized.")

        # Prepare parameters for the Anthropic messages.create() call
        kwargs = _build_anthropic_messages_kwargs(system_text, user_text, user_images, self.api_parameters)
        # Add timeout if not already specified
        if "timeout" not in kwargs:
            kwargs["timeout"] = 10 * 60  # 10 minutes

        # Send the request to Anthropic's API
        start_time = time.time()
        response = await self.client.messages.create(**kwargs)
        elapsed_time = time.time() - start_time
        
        # Extract token usage
        prompt_tokens = response.usage.input_tokens
        completion_tokens = response.usage.output_tokens
        total_tokens = prompt_tokens + completion_tokens
        
        LOG.debug(f"Anthropic request completed in {elapsed_time:.2f} seconds. Tokens: {total_tokens} (prompt: {prompt_tokens}, completion: {completion_tokens}).")
        VLM_LOG.debug(f"Token usage: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
        
        # Extract response text
        # Anthropic returns a list of content blocks
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_blocks)

    async def create_batch(self, requests: list[BatchRequest]) -> Batch:
        """Create a batch for processing multiple requests."""
        if not self.client:
            raise ConnectionError("Anthropic client is not initialized.")

        return AnthropicBatch(self.client, self.api_parameters, requests)
