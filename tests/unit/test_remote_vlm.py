from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from untitled.generator.remote_vlm import RemoteVLM
from untitled.generator.vlm_client import VLMClient
from untitled.generator.model_config import ModelConfig


@pytest.mark.asyncio
async def test_remote_vlm_start_is_noop():
    mock_config = MagicMock(spec=ModelConfig)
    client = RemoteVLM(mock_config, "http://remote-host", "token")

    assert client.state == VLMClient.State.STARTED
    await client.start()
    assert client.state == VLMClient.State.STARTED


@pytest.mark.asyncio
async def test_remote_vlm_stop_closes_connection_when_connected():
    mock_config = MagicMock(spec=ModelConfig)
    client = RemoteVLM(mock_config, "http://remote-host", "token")
    client.state = VLMClient.State.CONNECTED
    client.client = object()

    with patch.object(client, "close_connection", wraps=client.close_connection) as mock_close:
        await client.stop()

    mock_close.assert_awaited_once()
    assert client.state == VLMClient.State.STARTED
    assert client.client is None


def test_remote_vlm_parallelizable_and_remote_flags():
    mock_config = MagicMock(spec=ModelConfig)
    client = RemoteVLM(mock_config, "http://remote-host", "token")

    assert client.is_parallelizable() is True
    assert client.is_remote() is True
