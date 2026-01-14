from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from untitled.generator.remote_model import RemoteModel
from untitled.generator.model_client import ModelClient, ModelType
from untitled.generator.model_config import ModelConfig


@pytest.mark.asyncio
async def test_remote_model_start_is_noop():
    mock_config = MagicMock(spec=ModelConfig)
    client = RemoteModel(ModelType.VLM, 0, 8, "http://remote-host", "token")

    assert client.state == ModelClient.State.STARTED
    await client.start()
    assert client.state == ModelClient.State.STARTED


@pytest.mark.asyncio
async def test_remote_model_stop_closes_connection_when_connected():
    client = RemoteModel(ModelType.VLM, 0, 8, "http://remote-host", "token")
    client.state = ModelClient.State.CONNECTED
    client.client = object()

    with patch.object(client, "close_connection", wraps=client.close_connection) as mock_close:
        await client.stop()

    mock_close.assert_awaited_once()
    assert client.state == ModelClient.State.STARTED
    assert client.client is None


def test_remote_model_parallelizable_and_remote_flags():
    client = RemoteModel(ModelType.VLM, 0, 8, "http://remote-host", "token")

    assert client.is_parallelizable() is True
    assert client.is_remote() is True
