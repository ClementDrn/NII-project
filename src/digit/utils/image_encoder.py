from pathlib import Path
import io
from PIL import Image
import base64


def image_to_jpeg_base64(image_path: Path) -> str:
    """
    Encode an image to JPEG bytes.
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

    return base64_image
