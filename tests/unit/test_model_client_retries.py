from unittest.mock import AsyncMock, patch

import pytest

from untitled.generator.model_client import ModelClient, ModelType


class DummyModelClient(ModelClient):
    async def start(self):
        self.state = ModelClient.State.STARTED

    async def stop(self):
        self.state = ModelClient.State.DISABLED

    def is_remote(self):
        return False


@pytest.mark.asyncio
async def test_send_prompt_retries_on_timeout_then_succeeds():
    client = DummyModelClient(ModelType.LLM, 0, 1, "http://localhost:8000", "token", {"model": "demo"})

    timeout_error = Exception("litellm.Timeout: Connection timed out")
    success_response = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    with (
        patch("untitled.generator.model_client.acompletion", new=AsyncMock(side_effect=[timeout_error, success_response])) as mock_acompletion,
        patch("untitled.generator.model_client.asyncio.sleep", new=AsyncMock()) as mock_sleep,
        patch("untitled.generator.model_client.random.uniform", return_value=0.0),
    ):
        result = await client._send_prompt("sys", "user", [])

    assert result == "ok"
    assert mock_acompletion.await_count == 2
    mock_sleep.assert_awaited_once_with(client._RETRY_INITIAL_DELAY_SECONDS)


@pytest.mark.asyncio
async def test_send_prompt_retries_on_overload_until_exhausted():
    client = DummyModelClient(ModelType.LLM, 0, 1, "http://localhost:8000", "token", {"model": "demo"})

    overload_error = Exception("AnthropicError - overloaded_error: Overloaded")

    with (
        patch("untitled.generator.model_client.acompletion", new=AsyncMock(side_effect=overload_error)) as mock_acompletion,
        patch("untitled.generator.model_client.asyncio.sleep", new=AsyncMock()) as mock_sleep,
        patch("untitled.generator.model_client.random.uniform", return_value=0.0),
    ):
        with pytest.raises(Exception):
            await client._send_prompt("sys", "user", [])

    assert mock_acompletion.await_count == client._RETRY_MAX_ATTEMPTS + 1
    assert mock_sleep.await_count == client._RETRY_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_send_prompt_does_not_retry_non_retryable_error():
    client = DummyModelClient(ModelType.LLM, 0, 1, "http://localhost:8000", "token", {"model": "demo"})

    non_retryable_error = Exception("ValidationError: bad request")

    with (
        patch("untitled.generator.model_client.acompletion", new=AsyncMock(side_effect=non_retryable_error)) as mock_acompletion,
        patch("untitled.generator.model_client.asyncio.sleep", new=AsyncMock()) as mock_sleep,
    ):
        with pytest.raises(Exception):
            await client._send_prompt("sys", "user", [])

    assert mock_acompletion.await_count == 1
    mock_sleep.assert_not_awaited()
