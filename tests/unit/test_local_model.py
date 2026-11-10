from unittest.mock import create_autospec

import pytest

from digit.generator.local_model import LocalModel
from digit.generator.model_client import ModelClient, ModelType
from digit.generator.model_process import ModelProcess


@pytest.fixture
def process():
    return create_autospec(ModelProcess, instance=True, spec_set=True)


@pytest.fixture
def client(process, provider_adapter):
    return LocalModel(ModelType.VLM, process, 0, 8, provider_adapter)


@pytest.mark.asyncio
@pytest.mark.parametrize(('process_state', 'client_state'), [
    (ModelProcess.State.RUNNING, ModelClient.State.STARTED),
    (ModelProcess.State.STOPPED, ModelClient.State.DISABLED),
])
async def test_start_tracks_process_outcome(client, process, process_state, client_state):
    process.get_state.return_value = process_state
    await client.start()
    process.start.assert_awaited_once_with(owner=client)
    assert client.get_state() is client_state


@pytest.mark.asyncio
@pytest.mark.parametrize('connected', [False, True])
async def test_stop_releases_owned_process(client, process, provider_adapter, connected):
    client.state = ModelClient.State.CONNECTED if connected else ModelClient.State.STARTED
    process.get_state.return_value = ModelProcess.State.STOPPED
    await client.stop()
    process.stop.assert_awaited_once_with(owner=client)
    assert provider_adapter.close.await_count == int(connected)
    assert client.get_state() is ModelClient.State.DISABLED


@pytest.mark.asyncio
async def test_hot_client_signals_process_ownership(client, process):
    await client.set_hot()
    process.signal_ownership.assert_awaited_once_with(owner=client)


@pytest.mark.parametrize('parallel', [False, True])
def test_local_model_capabilities(client, process, parallel):
    process.is_parallel_to_simulator_supported.return_value = parallel
    assert client.is_parallelizable() is parallel
    assert client.is_remote() is False
