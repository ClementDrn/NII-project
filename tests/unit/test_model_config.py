import pytest

from untitled.generator.model_config import get_model_config, ModelConfig


def test_get_model_config_supported_model():
    config = get_model_config("r4b")

    assert isinstance(config, ModelConfig)
    assert config.model_name == "yannqi/R-4B"
    assert config.served_name == "r4b"
    assert config.limit_mm_per_prompt == {"image": 1}


def test_get_model_config_unsupported_model():
    with pytest.raises(ValueError) as exc_info:
        get_model_config("unknown")

    assert "Unsupported model" in str(exc_info.value)
