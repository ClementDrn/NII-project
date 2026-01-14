from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from untitled.generator.local_model import LocalModel
from untitled.generator.model_client import ModelClient, ModelType
from untitled.generator.model_process import ModelProcess


class DummyProcess:
    def __init__(self):
        self.terminate = MagicMock()
        self.kill = MagicMock()
        self.wait_calls = 0

    async def wait(self):
        self.wait_calls += 1
        return None


@pytest.mark.asyncio
async def test_local_model_start_success(monkeypatch):
    process = DummyProcess()

    create_proc_mock = AsyncMock(return_value=process)
    monkeypatch.setattr("untitled.generator.local_model.asyncio.create_subprocess_exec", create_proc_mock)

    sleep_mock = AsyncMock()
    monkeypatch.setattr("untitled.generator.local_model.asyncio.sleep", sleep_mock)
    
    # Create mock ModelProcess
    mock_model_process = MagicMock(spec=ModelProcess)
    mock_model_process.start = AsyncMock()
    mock_model_process.get_state = MagicMock(return_value=ModelProcess.State.RUNNING)
    
    client = LocalModel(ModelType.VLM, mock_model_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    client.test_connection = AsyncMock(return_value=True)

    await client.start()
    
    mock_model_process.start.assert_awaited_once()
    assert client.state == ModelClient.State.STARTED


@pytest.mark.asyncio
async def test_local_model_start_raises_when_connection_fails(monkeypatch):
    process = DummyProcess()
    create_proc_mock = AsyncMock(return_value=process)
    monkeypatch.setattr("untitled.generator.local_model.asyncio.create_subprocess_exec", create_proc_mock)

    sleep_mock = AsyncMock()
    monkeypatch.setattr("untitled.generator.local_model.asyncio.sleep", sleep_mock)
    
    # Create mock ModelProcess that fails to start
    mock_model_process = MagicMock(spec=ModelProcess)
    mock_model_process.start = AsyncMock()
    mock_model_process.get_state = MagicMock(return_value=ModelProcess.State.STOPPED)
    
    client = LocalModel(ModelType.VLM, mock_model_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    client.test_connection = AsyncMock(return_value=False)

    await client.start()
    assert client.state == ModelClient.State.DISABLED

@pytest.mark.asyncio
async def test_local_model_stop_terminates_process(monkeypatch):
    process = DummyProcess()
    
    async def wait_for_mock(coro, timeout):
        return await coro
    
    monkeypatch.setattr("untitled.generator.local_model.asyncio.wait_for", wait_for_mock)

    mock_model_process = MagicMock(spec=ModelProcess)
    mock_model_process.stop = AsyncMock()
    mock_model_process.get_state = MagicMock(return_value=ModelProcess.State.STOPPED)
    
    client = LocalModel(ModelType.VLM, mock_model_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    client.state = ModelClient.State.STARTED

    with patch.object(client, "close_connection", new=AsyncMock()) as mock_close:
        await client.stop()

    mock_close.assert_not_awaited()
    mock_model_process.stop.assert_awaited_once()
    assert client.state == ModelClient.State.DISABLED


@pytest.mark.asyncio
async def test_local_model_stop_closes_connection_when_connected(monkeypatch):
    process = DummyProcess()
    
    async def wait_for_mock(coro, timeout):
        return await coro
    
    monkeypatch.setattr("untitled.generator.local_model.asyncio.wait_for", wait_for_mock)

    mock_model_process = MagicMock(spec=ModelProcess)
    mock_model_process.stop = AsyncMock()
    mock_model_process.get_state = MagicMock(return_value=ModelProcess.State.STOPPED)
    
    client = LocalModel(ModelType.VLM, mock_model_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    client.state = ModelClient.State.CONNECTED
    client.client = object()

    with patch.object(client, "close_connection", new=AsyncMock(wraps=client.close_connection)) as mock_close:
        await client.stop()

    mock_close.assert_awaited_once()
    assert client.state == ModelClient.State.DISABLED


def test_local_model_is_not_remote():
    mock_model_process = MagicMock(spec=ModelProcess)
    client = LocalModel(ModelType.VLM, mock_model_process, 0, 8, "http://localhost:8000", "token")
    assert client.is_remote() is False
