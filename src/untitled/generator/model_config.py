from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Configuration for a specific model (LLM/VLM).
    These settings define model properties used when starting model servers.
    """
    model_name: str     # TODO: Should be optional, rename to huggingface_name
    served_name: str
    max_model_len: int = 8192
    max_prompt_len: int = -1     # -1 means no limit (other than max_model_len)
    reasoning_parser: str = ""
    thinking_activation_params: dict = None     # None means unsupported (will raise warning if thinking is enabled)
    thinking_deactivation_params: dict = None
    thinking_default_params: dict = field(default_factory=dict)
    limit_mm_per_prompt: dict = field(default_factory=dict)     # WARN: Deprecated TODO: Remove for release
    max_output_tokens_param: str = "max_tokens"  # Name of the parameter to set max output tokens
    temperature_param: str = "temperature"       # Name of the parameter to set temperature


def get_model_config(model_id: str) -> ModelConfig:
    """
    Get model configuration from model identifier.
    
    Args:
        model_id: Model identifier (e.g., "r4b")
        
    Returns:
        ModelConfig object with model properties
        
    Raises:
        ValueError: If model is not supported
    """
    models = {
        "r4b": ModelConfig(
            model_name="yannqi/r-4b",   # HuggingFace model name
            served_name="r4b",          # Should match model_name in config.yaml
            max_model_len=32768,        # Max for model instance context_window in config.yaml
            thinking_activation_params={
                "extra_body": {"chat_template_kwargs": {"thinking_mode": "long"}}
            },
            thinking_deactivation_params={
                "extra_body": {"chat_template_kwargs": {"thinking_mode": "short"}}
            },
            thinking_default_params={
                "extra_body": {"chat_template_kwargs": {"thinking_mode": "auto"}}
            }
        ),
        "qwen3-4b-thinking": ModelConfig(
            model_name="Qwen/qwen3-4b-thinking-2507",
            served_name="qwen3-4b-thinking",
            max_model_len=262144,
            reasoning_parser="deepseek_r1",
            thinking_activation_params={},      # enabled by default
            thinking_deactivation_params=None,  # not supported
            thinking_default_params={}
        ),
        "gpt-5-mini": ModelConfig(
            model_name="openai/gpt-5-mini",
            served_name="gpt-5-mini",     # FIXME: Not actually used for the API calls, right now it's the config.yaml's model_name that matters
            max_model_len=400000,
            thinking_activation_params={"reasoning_effort": "high"},
            thinking_deactivation_params={"reasoning_effort": "minimal"},
            thinking_default_params={},
            max_output_tokens_param="max_completion_tokens",  # Specific to OpenAI GPT-5 API
            temperature_param=""    # Unsupported
        ),
        "gpt-5.2": ModelConfig(
            model_name="openai/gpt-5.2",
            served_name="gpt-5.2",
            max_model_len=400000,
            thinking_activation_params={"reasoning_effort": "medium"},
            thinking_deactivation_params={"reasoning_effort": "none"},
            thinking_default_params={},
            max_output_tokens_param="max_completion_tokens",  # Specific to OpenAI GPT-5 API
            temperature_param=""    # Unsupported (temperature is only supported when reasoning effort is "none")
        ),
        "gemini-3.1-pro-preview": ModelConfig(
            model_name="gemini/gemini-3.1-pro-preview",
            served_name="gemini-3.1-pro-preview",
            max_model_len=1000000,
            thinking_activation_params={"reasoning_effort": "high"},
            thinking_deactivation_params={"reasoning_effort": "low"},
            thinking_default_params={},
            max_output_tokens_param="max_output_tokens",  # Specific to Gemini 3.1 Pro API
            temperature_param="temperature"    # Supported in all reasoning effort modes for Gemini
        ),
        "claude-opus-4.6": ModelConfig(
            model_name="anthropic/claude-opus-4-6",
            served_name="claude-opus-4.6",
            max_model_len=1000000,
            thinking_activation_params={"reasoning_effort": "high"},
            thinking_deactivation_params={"reasoning_effort": "low"},
            thinking_default_params={},
            max_output_tokens_param="max_tokens",  # Specific to Claude Opus API
            temperature_param=""    # Not supported when thinking is enabled
        ),
        # Add more models here as needed
    }
    
    if model_id not in models:
        raise ValueError(f"Unsupported model: {model_id}. Supported models: {list(models.keys())}")
    
    return models[model_id]
