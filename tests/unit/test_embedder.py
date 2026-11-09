from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

from digit.filter.embedder import DINOEmbedder, Embedder


def test_embedder_requires_implementation():
    with pytest.raises(TypeError):
        Embedder()


def test_dino_pools_tokens_normalizes_and_returns_cpu_vector(tmp_path, monkeypatch):
    # Asymmetric tokens distinguish mean pooling from selecting the first token.
    hidden_state = torch.tensor([[[3.0, 0.0], [0.0, 4.0]]])
    image_path = tmp_path / 'image.png'
    Image.new('L', (3, 2), color=100).save(image_path)
    inputs = MagicMock()
    inputs.to.return_value = {'pixel_values': torch.ones(1, 3, 2, 3)}
    processor = MagicMock(return_value=inputs)
    model = MagicMock(return_value=SimpleNamespace(last_hidden_state=hidden_state))
    model.to.return_value = model
    load_processor = MagicMock(return_value=processor)
    load_model = MagicMock(return_value=model)
    monkeypatch.setattr('digit.filter.embedder.AutoImageProcessor.from_pretrained', load_processor)
    monkeypatch.setattr('digit.filter.embedder.AutoModel.from_pretrained', load_model)

    embedder = DINOEmbedder()
    result = embedder.embed(image_path)

    load_processor.assert_called_once_with('facebook/dinov2-base')
    load_model.assert_called_once_with('facebook/dinov2-base')
    processor.assert_called_once()
    assert processor.call_args.kwargs['images'].mode == 'RGB'
    assert processor.call_args.kwargs['return_tensors'] == 'pt'
    model.assert_called_once_with(pixel_values=inputs.to.return_value['pixel_values'])
    torch.testing.assert_close(result, torch.tensor([0.6, 0.8]))
    assert result.device == torch.device('cpu')
