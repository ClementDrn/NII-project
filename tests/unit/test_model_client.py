"""Model lifecycle and request delegation at the provider boundary."""
from unittest.mock import create_autospec

import pytest

from digit.generator.batch import Batch, BatchRequest
from digit.generator.model_client import ModelClient, ModelType, parse_model_response
from digit.generator.request import Request, TextBlock


class ConcreteModelClient(ModelClient):
    async def start(self):
        self.state = self.State.STARTED

    async def stop(self):
        self.state = self.State.DISABLED

    def is_remote(self):
        return False


@pytest.fixture
def client(provider_adapter):
    return ConcreteModelClient(ModelType.VLM, 0, 8, provider_adapter)


@pytest.mark.parametrize(('response', 'expected'), [
    ('<think>details</think>\n\nAnswer', 'Answer'),
    ('Final output', 'Final output'),
    ('  <think>details</think>\n\n  Answer me  ', 'Answer me'),
    ('<think>first</think>\n\nAnswer with </think> literal', 'Answer with </think> literal'),
])
def test_parse_model_response(response, expected):
    assert parse_model_response(response) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('state', [ModelClient.State.DISABLED, ModelClient.State.STARTED])
async def test_requests_require_connection(client, provider_adapter, state):
    client.state = state
    with pytest.raises(ConnectionError, match='not connected'):
        await client.prompt(Request('system', []))
    with pytest.raises(ConnectionError, match='not connected'):
        await client.batch([])
    provider_adapter.send_prompt.assert_not_awaited()
    provider_adapter.create_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_open_connection_starts_and_verifies_provider(client, provider_adapter):
    await client.open_connection()
    provider_adapter.open.assert_awaited_once_with()
    provider_adapter.test_connection.assert_awaited_once_with()
    assert client.get_state() is ModelClient.State.CONNECTED


@pytest.mark.asyncio
async def test_failed_connection_does_not_mark_client_connected(client, provider_adapter):
    provider_adapter.test_connection.return_value = False
    with pytest.raises(ConnectionError, match='Failed to connect'):
        await client.open_connection()
    assert client.get_state() is ModelClient.State.STARTED


@pytest.mark.asyncio
async def test_prompt_delegates_original_request(client, provider_adapter):
    await client.open_connection()
    request = Request('system', [TextBlock('Turn left')])
    provider_adapter.send_prompt.return_value = 'result'
    assert await client.prompt(request) == 'result'
    provider_adapter.send_prompt.assert_awaited_once_with(request)


@pytest.mark.asyncio
async def test_provider_error_propagates(client, provider_adapter):
    await client.open_connection()
    error = ValueError('invalid request')
    provider_adapter.send_prompt.side_effect = error
    with pytest.raises(ValueError, match='invalid request') as exc:
        await client.prompt(Request('', []))
    assert exc.value is error


@pytest.mark.asyncio
async def test_close_connection_closes_provider(client, provider_adapter):
    await client.open_connection()
    await client.close_connection()
    provider_adapter.close.assert_awaited_once_with()
    assert client.get_state() is ModelClient.State.STARTED


@pytest.mark.asyncio
async def test_batch_delegates_requests(client, provider_adapter):
    await client.open_connection()
    batch = create_autospec(Batch, instance=True)
    provider_adapter.create_batch.return_value = batch
    requests = [BatchRequest('request-1', 'system', [TextBlock('Turn left')])]
    assert await client.batch(requests) is batch
    provider_adapter.create_batch.assert_awaited_once_with(requests)


@pytest.mark.asyncio
async def test_unsupported_batch_raises(client, provider_adapter):
    await client.open_connection()
    provider_adapter.create_batch.return_value = None
    with pytest.raises(RuntimeError, match='Failed to create batch'):
        await client.batch([])
