from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific VLM model."""
    model_name: str
    served_name: str
    max_model_len: int = 8192
    limit_mm_per_prompt: dict = None
    
    # def __post_init__(self):
    #     if self.limit_mm_per_prompt is None:
    #         self.limit_mm_per_prompt = {"image": 1}


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
            model_name="yannqi/R-4B",
            served_name="r4b",
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1}
        ),
        # Add more models here as needed
    }
    
    if model_id not in models:
        raise ValueError(f"Unsupported model: {model_id}. Supported models: {list(models.keys())}")
    
    return models[model_id]
