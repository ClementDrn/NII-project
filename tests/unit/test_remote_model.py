import pytest

from digit.generator.remote_model import RemoteModel
from digit.generator.model_client import ModelClient, ModelType


@pytest.fixture
def client(provider_adapter):
    return RemoteModel(ModelType.VLM, 0, 8, provider_adapter)


@pytest.mark.asyncio
async def test_start_assumes_remote_server_running(client):
    assert client.get_state() is ModelClient.State.STARTED
    await client.start()
    assert client.get_state() is ModelClient.State.STARTED


@pytest.mark.asyncio
@pytest.mark.parametrize('connected', [False, True])
async def test_stop_closes_only_connected_provider(client, provider_adapter, connected):
    if connected:
        await client.open_connection()
    await client.stop()
    assert provider_adapter.close.await_count == int(connected)
    assert client.get_state() is ModelClient.State.STARTED


def test_remote_capabilities(client):
    assert client.is_parallelizable() is True
    assert client.is_remote() is True
