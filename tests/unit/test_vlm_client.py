import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest
import pytest_asyncio
from PIL import Image

from untitled.generator.vlm_client import (
    VLMClient,
    encode_image_to_base64,
    parse_vlm_response,
)
from untitled.generator.model_config import ModelConfig


class DummyVLMClient(VLMClient):
    """Concrete implementation for testing abstract VLMClient behaviour."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.started = False
        self.stopped = False

    async def start(self):  # pragma: no cover - behaviour verified via state checks
        self.started = True
        self.state = VLMClient.State.STARTED

    async def stop(self):  # pragma: no cover - behaviour verified via state checks
        self.stopped = True
        self.state = VLMClient.State.DISABLED

    def is_remote(self):  # pragma: no cover - not relevant for these tests
        return False


@pytest_asyncio.fixture(scope="function")
async def dummy_client():
    client = DummyVLMClient(0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    await client.start()
    return client


def test_encode_image_to_base64(tmp_path):
    image_path = tmp_path / "test.jpg"
    Image.new("RGB", (2, 2), color="red").save(image_path, format="JPEG")

    encoded = encode_image_to_base64(image_path)

    assert encoded.startswith("data:image/jpeg;base64,")
    payload = encoded.split(",", 1)[1]
    # Ensure payload decodes without raising
    base64.b64decode(payload)


@patch("untitled.generator.vlm_client.Image.open")
def test_encode_image_to_base64_uses_context_manager(mock_image_open):
    mock_image = MagicMock()

    def fake_save(buffer, format):
        assert format == "JPEG"
        buffer.write(b"bytes")

    mock_image.save.side_effect = fake_save
    mock_image_open.return_value.__enter__ = MagicMock(return_value=mock_image)
    mock_image_open.return_value.__exit__ = MagicMock(return_value=False)

    result = encode_image_to_base64(Path("/tmp/example.jpg"))

    assert result.startswith("data:image/jpeg;base64,")
    mock_image_open.assert_called_once_with(Path("/tmp/example.jpg"))
    mock_image.save.assert_called_once()


def test_parse_vlm_response_with_reasoning():
    response = "<think>details</think>\n\nAnswer"
    assert parse_vlm_response(response) == "Answer"


def test_parse_vlm_response_without_reasoning():
    response = "Final output"
    assert parse_vlm_response(response) == "Final output"


def test_parse_vlm_response_strips_whitespace():
    response = "  <think>details</think>\n\n  Answer me  "
    assert parse_vlm_response(response) == "Answer me"


def test_parse_vlm_response_multiple_think_blocks():
    response = "<think>first</think>\n\nAnswer with </think> literal"
    assert parse_vlm_response(response) == "Answer with </think> literal"


@pytest.mark.asyncio
async def test_prompt_requires_connection(dummy_client):
    dummy_client.state = VLMClient.State.STARTED
    with pytest.raises(ConnectionError):
        await dummy_client.prompt("sys", "user", [])


@pytest.mark.asyncio
async def test_prompt_returns_result(dummy_client):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "result"

    dummy_client.state = VLMClient.State.CONNECTED
    dummy_client.client = MagicMock()
    dummy_client.client.chat.completions.create = AsyncMock(return_value=response)

    result = await dummy_client.prompt("sys", "user", None)
    assert result == "result"
    dummy_client.client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_prompt_passes_api_parameters(dummy_client):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "ok"

    dummy_client.state = VLMClient.State.CONNECTED
    dummy_client.client = MagicMock()
    dummy_client.client.chat.completions.create = AsyncMock(return_value=response)

    dummy_client.api_parameters = {"model": "model-x", "temperature": 0.2, "max_tokens": 12}

    result = await dummy_client.prompt("sys", "user", None)
    assert result == "ok"

    await_args = dummy_client.client.chat.completions.create.await_args
    assert await_args.kwargs["model"] == "model-x"
    assert await_args.kwargs["temperature"] == 0.2
    assert await_args.kwargs["max_tokens"] == 12


@pytest.mark.asyncio
async def test_open_connection_initialises_async_openai(dummy_client):
    dummy_client.test_connection = AsyncMock(return_value=True)

    async_openai_instance = MagicMock()
    with patch("untitled.generator.vlm_client.AsyncOpenAI", return_value=async_openai_instance) as mock_async_openai:
        await dummy_client.open_connection()

    mock_async_openai.assert_called_once_with(api_key="token", base_url="http://localhost:8000/v1")
    dummy_client.test_connection.assert_awaited_once()
    assert dummy_client.client is async_openai_instance
    assert dummy_client.state == VLMClient.State.CONNECTED


@pytest.mark.asyncio
async def test_open_connection_raises_when_test_connection_fails(dummy_client):
    dummy_client.test_connection = AsyncMock(return_value=False)

    with patch("untitled.generator.vlm_client.AsyncOpenAI", return_value=MagicMock()):
        with pytest.raises(ConnectionError):
            await dummy_client.open_connection()

    assert dummy_client.state == VLMClient.State.STARTED


@pytest.mark.asyncio
async def test_close_connection_resets_state(dummy_client):
    dummy_client.state = VLMClient.State.CONNECTED
    dummy_client.client = MagicMock()

    await dummy_client.close_connection()

    assert dummy_client.client is None
    assert dummy_client.state == VLMClient.State.STARTED


@pytest.mark.asyncio
async def test_test_connection_warns_when_not_connected(dummy_client):
    dummy_client.state = VLMClient.State.STARTED
    assert not await dummy_client.test_connection()


@pytest.mark.asyncio
async def test_test_connection_makes_request_when_connected(dummy_client):
    dummy_client.state = VLMClient.State.CONNECTED
    response = MagicMock()
    response.choices = [MagicMock()]

    mock_async = AsyncMock(return_value=response)
    dummy_client.client = MagicMock()
    dummy_client.client.chat.completions.create = mock_async

    assert await dummy_client.test_connection()
    mock_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_send_prompt_includes_images_and_text(dummy_client):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "resp"

    dummy_client.client = MagicMock()
    dummy_client.client.chat.completions.create = AsyncMock(return_value=response)

    with patch("untitled.generator.vlm_client.encode_image_to_base64", side_effect=["img1", "img2"]) as mock_encode:
        result = await dummy_client._send_prompt("sys", "user", [Path("img1.jpg"), Path("img2.jpg")])

    assert result == "resp"
    assert mock_encode.call_args_list == [call(Path("img1.jpg")), call(Path("img2.jpg"))]

    call_args = dummy_client.client.chat.completions.create.await_args
    messages = call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "sys"}
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]
    assert content[0] == {"type": "image_url", "image_url": {"url": "img1"}}
    assert content[1] == {"type": "image_url", "image_url": {"url": "img2"}}
    assert content[2] == {"type": "text", "text": "user"}


@pytest.mark.asyncio
async def test_send_prompt_handles_none_text(dummy_client):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "resp"

    dummy_client.client = MagicMock()
    dummy_client.client.chat.completions.create = AsyncMock(return_value=response)

    with patch("untitled.generator.vlm_client.encode_image_to_base64", return_value="img-data"):
        await dummy_client._send_prompt("sys", None, [Path("img.jpg")])

    call_args = dummy_client.client.chat.completions.create.await_args
    content = call_args.kwargs["messages"][1]["content"]
    assert len(content) == 1
    assert content[0] == {"type": "image_url", "image_url": {"url": "img-data"}}


@pytest.mark.asyncio
async def test_send_prompt_measures_time(dummy_client):
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "resp"

    dummy_client.client = MagicMock()
    dummy_client.client.chat.completions.create = AsyncMock(return_value=response)

    call_count = [0]
    def mock_time_func():
        call_count[0] += 1
        return 1.0 + call_count[0] * 0.5
    
    with patch("untitled.generator.vlm_client.time.time", side_effect=mock_time_func) as mock_time:
        await dummy_client._send_prompt("sys", "user", None)

    assert mock_time.call_count >= 2
