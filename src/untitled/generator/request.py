from dataclasses import dataclass
from typing import List, Literal
from pathlib import Path


@dataclass
class TextBlock:
    text: str
    to_be_cached: bool = False
    type: Literal["text"] = "text"

@dataclass
class ImageBlock:
    image_path: Path
    to_be_cached: bool = False
    type: Literal["image"] = "image"

# Union type for request content blocks
RequestContentBlock = TextBlock | ImageBlock

@dataclass
class Request:
    """Represents a request to be sent to a LLM/VLM."""

    system_text: str
    user_content: List[RequestContentBlock]
