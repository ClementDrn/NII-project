"""Retries belong to the provider adapter, not ModelClient."""
from unittest.mock import AsyncMock, call

import pytest

from digit.generator.providers.litellm import LiteLLMAdapter
from digit.generator.request import Request, TextBlock


@pytest.fixture
def adapter():
    return LiteLLMAdapter('http://localhost:8000', 'test-token', {'model': 'openai/demo'})


@pytest.fixture
def completion(monkeypatch):
    completion = AsyncMock()
    monkeypatch.setattr('digit.generator.providers.litellm.acompletion', completion)
    return completion


@pytest.fixture
def retry_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr('digit.generator.providers.litellm.asyncio.sleep', sleep)
    monkeypatch.setattr('digit.generator.providers.litellm.random.uniform', lambda *args: 0.0)
    return sleep


@pytest.mark.asyncio
async def test_timeout_retries_same_request_then_returns_response(adapter, completion, retry_sleep):
    completion.side_effect = [TimeoutError('Connection timed out'),
                              {'choices': [{'message': {'content': 'ok'}}]}]
    assert await adapter.send_prompt(Request('sys', [TextBlock('user')])) == 'ok'
    assert completion.await_count == 2
    assert completion.await_args_list[0] == completion.await_args_list[1]
    retry_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_overload_exhausts_retries_with_capped_backoff(adapter, completion, retry_sleep):
    error = RuntimeError('overloaded_error: Overloaded')
    completion.side_effect = error
    with pytest.raises(RuntimeError, match='Overloaded') as exc:
        await adapter.send_prompt(Request('sys', []))
    assert exc.value is error
    assert completion.await_count == 11
    assert retry_sleep.await_args_list == [call(delay) for delay in
                                          [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]]


@pytest.mark.asyncio
async def test_invalid_request_propagates_without_retry(adapter, completion, retry_sleep):
    error = ValueError('bad request')
    completion.side_effect = error
    with pytest.raises(ValueError, match='bad request') as exc:
        await adapter.send_prompt(Request('sys', []))
    assert exc.value is error
    assert completion.await_count == 1
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_multimodal_request_preserves_content_order_and_parameters(adapter, completion, tmp_path):
    import base64
    import io
    from PIL import Image
    from digit.generator.request import ImageBlock

    image_path = tmp_path / 'frame.png'
    Image.new('RGB', (3, 2), color='red').save(image_path)
    adapter.api_parameters.update({'temperature': 0.2, 'max_tokens': 12})
    completion.return_value = {'choices': [{'message': {'content': 'answer'}}]}
    result = await adapter.send_prompt(Request('system', [
        TextBlock('before'), ImageBlock(image_path), TextBlock('after'),
    ]))
    assert result == 'answer'
    kwargs = completion.await_args.kwargs
    assert kwargs['model'] == 'openai/demo'
    assert kwargs['api_base'] == 'http://localhost:8000/v1'
    assert kwargs['api_key'] == 'test-token'
    assert kwargs['temperature'] == 0.2
    assert kwargs['max_tokens'] == 12
    assert kwargs['messages'][0] == {'role': 'system', 'content': 'system'}
    content = kwargs['messages'][1]['content']
    assert content[0] == {'type': 'text', 'text': 'before'}
    assert content[2] == {'type': 'text', 'text': 'after'}
    prefix, payload = content[1]['image_url']['url'].split(',', 1)
    assert prefix == 'data:image/jpeg;base64'
    with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as image:
        assert image.format == 'JPEG'
        assert image.size == (3, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('response', [
    {'choices': [{'message': {'content': 'plain text'}}]},
    {'choices': [{'message': {'content': [{'type': 'text', 'text': 'plain text'}]}}]},
])
async def test_response_formats_return_text(adapter, completion, response):
    completion.return_value = response
    assert await adapter.send_prompt(Request('', [TextBlock('user')])) == 'plain text'
    assert completion.await_args.kwargs['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'user'}]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize('token_parameter', ['max_tokens', 'max_completion_tokens', 'max_output_tokens'])
async def test_connection_probe_uses_small_budget_without_mutating_config(adapter, completion, token_parameter):
    from types import SimpleNamespace

    adapter.api_parameters = {'model': 'openai/demo', token_parameter: 500}
    completion.return_value = SimpleNamespace(choices=[object()])
    assert await adapter.test_connection()
    kwargs = completion.await_args.kwargs
    assert kwargs[token_parameter] == 5
    assert adapter.api_parameters[token_parameter] == 500
    assert {key for key in kwargs if key.startswith('max_')} == {token_parameter}
