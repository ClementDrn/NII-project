import base64
import io

import pytest
from PIL import Image

from digit.utils.image_encoder import image_to_jpeg_base64


@pytest.mark.parametrize('source_format', ['PNG', 'JPEG'])
def test_image_is_encoded_as_jpeg(tmp_path, source_format):
    path = tmp_path / 'image'
    Image.new('RGB', (3, 2), color='red').save(path, format=source_format)
    encoded = image_to_jpeg_base64(path)
    with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as decoded:
        assert decoded.format == 'JPEG'
        assert decoded.size == (3, 2)
        red, green, blue = decoded.getpixel((0, 0))
        assert red > 240 and green < 10 and blue < 10


def test_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        image_to_jpeg_base64(tmp_path / 'missing.jpg')
