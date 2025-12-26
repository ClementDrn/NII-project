from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from untitled.generator.local_vlm import LocalVLM
from untitled.generator.vlm_client import VLMClient
from untitled.generator.vlm_process import VLMProcess


class DummyProcess:
    def __init__(self):
        self.terminate = MagicMock()
        self.kill = MagicMock()
        self.wait_calls = 0

    async def wait(self):
        self.wait_calls += 1
        return None


@pytest.mark.asyncio
async def test_local_vlm_start_success(monkeypatch):
    process = DummyProcess()

    create_proc_mock = AsyncMock(return_value=process)
    monkeypatch.setattr("untitled.generator.local_vlm.asyncio.create_subprocess_exec", create_proc_mock)

    sleep_mock = AsyncMock()
    monkeypatch.setattr("untitled.generator.local_vlm.asyncio.sleep", sleep_mock)
    
    # Create mock VLMProcess
    mock_vlm_process = MagicMock(spec=VLMProcess)
    mock_vlm_process.start = AsyncMock()
    mock_vlm_process.get_state = MagicMock(return_value=VLMProcess.State.RUNNING)
    
    vlm = LocalVLM(mock_vlm_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    vlm.test_connection = AsyncMock(return_value=True)

    await vlm.start()
    
    mock_vlm_process.start.assert_awaited_once()
    assert vlm.state == VLMClient.State.STARTED


@pytest.mark.asyncio
async def test_local_vlm_start_raises_when_connection_fails(monkeypatch):
    process = DummyProcess()
    create_proc_mock = AsyncMock(return_value=process)
    monkeypatch.setattr("untitled.generator.local_vlm.asyncio.create_subprocess_exec", create_proc_mock)

    sleep_mock = AsyncMock()
    monkeypatch.setattr("untitled.generator.local_vlm.asyncio.sleep", sleep_mock)
    
    # Create mock VLMProcess that fails to start
    mock_vlm_process = MagicMock(spec=VLMProcess)
    mock_vlm_process.start = AsyncMock()
    mock_vlm_process.get_state = MagicMock(return_value=VLMProcess.State.STOPPED)
    
    vlm = LocalVLM(mock_vlm_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    vlm.test_connection = AsyncMock(return_value=False)

    await vlm.start()
    assert vlm.state == VLMClient.State.DISABLED
@pytest.mark.asyncio
async def test_local_vlm_stop_terminates_process(monkeypatch):
    process = DummyProcess()
    
    async def wait_for_mock(coro, timeout):
        return await coro
    
    monkeypatch.setattr("untitled.generator.local_vlm.asyncio.wait_for", wait_for_mock)

    mock_vlm_process = MagicMock(spec=VLMProcess)
    mock_vlm_process.stop = AsyncMock()
    mock_vlm_process.get_state = MagicMock(return_value=VLMProcess.State.STOPPED)
    
    vlm = LocalVLM(mock_vlm_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    vlm.state = VLMClient.State.STARTED

    with patch.object(vlm, "close_connection", new=AsyncMock()) as mock_close:
        await vlm.stop()

    mock_close.assert_not_awaited()
    mock_vlm_process.stop.assert_awaited_once()
    assert vlm.state == VLMClient.State.DISABLED


@pytest.mark.asyncio
async def test_local_vlm_stop_closes_connection_when_connected(monkeypatch):
    process = DummyProcess()
    
    async def wait_for_mock(coro, timeout):
        return await coro
    
    monkeypatch.setattr("untitled.generator.local_vlm.asyncio.wait_for", wait_for_mock)

    mock_vlm_process = MagicMock(spec=VLMProcess)
    mock_vlm_process.stop = AsyncMock()
    mock_vlm_process.get_state = MagicMock(return_value=VLMProcess.State.STOPPED)
    
    vlm = LocalVLM(mock_vlm_process, 0, 8, "http://localhost:8000", "token", {"model": "r4b"})
    vlm.state = VLMClient.State.CONNECTED
    vlm.client = object()

    with patch.object(vlm, "close_connection", new=AsyncMock(wraps=vlm.close_connection)) as mock_close:
        await vlm.stop()

    mock_close.assert_awaited_once()
    assert vlm.state == VLMClient.State.DISABLED


def test_local_vlm_is_not_remote():
    mock_vlm_process = MagicMock(spec=VLMProcess)
    vlm = LocalVLM(mock_vlm_process, 0, 8, "http://localhost:8000", "token")
    assert vlm.is_remote() is False
