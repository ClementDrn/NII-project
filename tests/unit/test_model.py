from unittest.mock import create_autospec

import pytest

from digit.core.base import CONFIG
from digit.core.settings import HostConfig, ModelInstanceConfig, LocalVLMConfig, RemoteVLMConfig
from digit.generator.local_model import LocalModel
from digit.generator.remote_model import RemoteModel
from digit.generator.model import ModelClientFactory, create_model_client
from digit.generator.model_client import ModelType
from digit.generator.model_process import ModelProcess
from digit.generator.providers.anthropic import AnthropicAdapter
from digit.generator.providers.litellm import LiteLLMAdapter


@pytest.fixture(autouse=True)
def isolated_process_cache(monkeypatch):
    monkeypatch.setattr(ModelClientFactory, '_model_processes', {})


def test_local_clients_reuse_process_and_preserve_adapter_parameters(monkeypatch):
    process_factory = create_autospec(ModelProcess)
    monkeypatch.setattr('digit.generator.model.ModelProcess', process_factory)
    config = ModelInstanceConfig(id='local', model_name='r4b', type='vlm', priority=2,
                                 batch_size=4, local_config=LocalVLMConfig(port=8123))
    parameters = {'model': 'r4b', 'temperature': 0.2}
    first = create_model_client(config, parameters)
    second = create_model_client(config, parameters)
    assert isinstance(first, LocalModel)
    assert first is not second
    process_factory.assert_called_once()
    assert process_factory.call_args.args[0] == 'http://localhost:8123'
    assert first._process is second._process is process_factory.return_value
    assert first.priority == 2
    assert first.batch_size == 4
    assert isinstance(first.adapter, LiteLLMAdapter)
    assert first.adapter.host == 'http://localhost:8123'
    assert first.adapter.api_key == 'EMPTY'
    assert first.adapter.api_parameters == parameters


@pytest.mark.parametrize(('host', 'adapter_type'), [
    ('https://api.anthropic.com', AnthropicAdapter),
    ('https://api.openai.com', LiteLLMAdapter),
    ('http://private-model:8000', LiteLLMAdapter),
])
def test_remote_client_uses_configured_host_and_provider(monkeypatch, host, adapter_type):
    monkeypatch.setattr(CONFIG, 'hosts', [HostConfig(id='test-host', host=host, api_key='test-token')])
    config = ModelInstanceConfig(id='remote', model_name='r4b', type='llm', priority=3,
                                 batch_size=2, remote_config=RemoteVLMConfig(host_id='test-host'))
    parameters = {'model': 'r4b', 'max_tokens': 12}
    client = create_model_client(config, parameters)
    assert isinstance(client, RemoteModel)
    assert client.model_type is ModelType.LLM
    assert client.priority == 3
    assert client.batch_size == 2
    assert isinstance(client.adapter, adapter_type)
    assert client.adapter.api_key == 'test-token'
    assert client.adapter.api_parameters == parameters


@pytest.mark.parametrize('both', [False, True])
def test_ambiguous_or_missing_location_does_not_create_client(both):
    config = ModelInstanceConfig(id='invalid', model_name='r4b', type='vlm',
        local_config=LocalVLMConfig() if both else None,
        remote_config=RemoteVLMConfig(host_id='test-host') if both else None)
    assert create_model_client(config) is None
