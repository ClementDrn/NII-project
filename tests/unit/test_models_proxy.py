from untitled.core.settings import HostConfig, LocalVLMConfig, ModelInstanceConfig, RemoteVLMConfig, Settings
from untitled.generator.models_proxy import ModelProxyManager


def test_resolve_upstream_api_base_appends_v1_for_openai_compatible_models():
    manager = ModelProxyManager()

    assert manager._resolve_upstream_api_base("gpt-5-mini", "http://example.com") == "http://example.com/v1"


def test_resolve_upstream_api_base_preserves_explicit_path():
    manager = ModelProxyManager()

    assert manager._resolve_upstream_api_base("gpt-5-mini", "http://example.com/api") == "http://example.com/api"


def test_resolve_upstream_api_base_keeps_provider_native_root():
    manager = ModelProxyManager()

    assert manager._resolve_upstream_api_base("gemini-3.1-pro-preview", "https://example.com") == "https://example.com"


def test_build_model_entries_uses_proxy_url_resolution_for_local_and_remote_models():
    manager = ModelProxyManager()
    settings = Settings(
        hosts=[
            HostConfig(id="openai", host="http://remote.example.com", api_key="secret"),
        ],
    )
    settings.active_model_instances = [
        ModelInstanceConfig(
            id="local-1",
            model_name="r4b",
            type="vlm",
            local_config=LocalVLMConfig(port=8001),
        ),
        ModelInstanceConfig(
            id="remote-1",
            model_name="gpt-5-mini",
            type="vlm",
            remote_config=RemoteVLMConfig(host_id="openai"),
        ),
    ]

    entries = manager._build_model_entries(settings)

    assert len(entries) == 2
    assert entries[0].api_base == "http://127.0.0.1:8001/v1"
    assert entries[1].api_base == "http://remote.example.com/v1"