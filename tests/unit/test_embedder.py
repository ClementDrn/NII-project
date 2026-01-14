import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import torch

from untitled.filter.embedder import Embedder, DINOEmbedder


class TestEmbedderInterface:
    """Tests for the Embedder abstract base class interface."""

    def test_embedder_is_abstract(self):
        """Test that Embedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Embedder()

    def test_embedder_requires_embed_method(self):
        """Test that concrete implementations must implement the embed method."""
        class IncompleteEmbedder(Embedder):
            pass

        with pytest.raises(TypeError):
            IncompleteEmbedder()

    def test_embedder_concrete_implementation_works(self):
        """Test that a proper concrete implementation can be instantiated."""
        class ConcreteEmbedder(Embedder):
            def embed(self, image_path: Path) -> torch.Tensor:
                return torch.zeros(768)

        embedder = ConcreteEmbedder()
        assert embedder is not None

    def test_embed_method_signature(self):
        """Test that embed method accepts Path and returns Tensor."""
        class ConcreteEmbedder(Embedder):
            def embed(self, image_path: Path) -> torch.Tensor:
                return torch.randn(768)

        embedder = ConcreteEmbedder()
        result = embedder.embed(Path("/fake/path/image.png"))
        assert isinstance(result, torch.Tensor)


class TestDINOEmbedder:
    """Tests for the DINOEmbedder class."""

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    def test_dino_embedder_initialization(self, mock_processor, mock_model):
        """Test that DINOEmbedder initializes with correct model and processor."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()

        embedder = DINOEmbedder()

        mock_processor.from_pretrained.assert_called_once_with('facebook/dinov2-base')
        mock_model.from_pretrained.assert_called_once_with('facebook/dinov2-base')
        assert embedder is not None

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    def test_dino_embedder_is_embedder_subclass(self, mock_processor, mock_model):
        """Test that DINOEmbedder is a subclass of Embedder."""
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = MagicMock()

        embedder = DINOEmbedder()

        assert isinstance(embedder, Embedder)

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_returns_tensor(self, mock_image, mock_processor, mock_model):
        """Test that embed method returns a torch.Tensor."""
        # Setup mocks
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_image.open.return_value.convert.return_value = MagicMock()

        embedder = DINOEmbedder()
        result = embedder.embed(Path("/fake/path/image.png"))

        assert isinstance(result, torch.Tensor)

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_returns_1d_tensor(self, mock_image, mock_processor, mock_model):
        """Test that embed method returns a 1-dimensional tensor (embedding vector)."""
        # Setup mocks
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_image.open.return_value.convert.return_value = MagicMock()

        embedder = DINOEmbedder()
        result = embedder.embed(Path("/fake/path/image.png"))

        assert result.dim() == 1

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_returns_normalized_tensor(self, mock_image, mock_processor, mock_model):
        """Test that embed method returns a normalized tensor (unit vector)."""
        # Setup mocks
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_image.open.return_value.convert.return_value = MagicMock()

        embedder = DINOEmbedder()
        result = embedder.embed(Path("/fake/path/image.png"))

        # Check that the tensor is normalized (magnitude should be ~1)
        norm = torch.norm(result).item()
        assert abs(norm - 1.0) < 0.01

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_opens_image_file(self, mock_image, mock_processor, mock_model):
        """Test that embed method opens the image file at the given path."""
        # Setup mocks
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_image.open.return_value.convert.return_value = MagicMock()

        embedder = DINOEmbedder()
        test_path = Path("/fake/path/image.png")
        embedder.embed(test_path)

        mock_image.open.assert_called_once_with(test_path)

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_converts_image_to_rgb(self, mock_image, mock_processor, mock_model):
        """Test that embed method converts the image to RGB format."""
        # Setup mocks
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_image.open.return_value.convert.return_value = MagicMock()

        embedder = DINOEmbedder()
        embedder.embed(Path("/fake/path/image.png"))

        mock_image.open.return_value.convert.assert_called_once_with("RGB")

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_returns_cpu_tensor(self, mock_image, mock_processor, mock_model):
        """Test that embed method returns a tensor on CPU."""
        # Setup mocks
        mock_processor.from_pretrained.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_image.open.return_value.convert.return_value = MagicMock()

        embedder = DINOEmbedder()
        result = embedder.embed(Path("/fake/path/image.png"))

        assert result.device == torch.device('cpu')

    @patch('untitled.filter.embedder.AutoModel')
    @patch('untitled.filter.embedder.AutoImageProcessor')
    @patch('untitled.filter.embedder.Image')
    def test_embed_uses_processor(self, mock_image, mock_processor, mock_model):
        """Test that embed method uses the image processor."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.from_pretrained.return_value = mock_processor_instance
        mock_model_instance = MagicMock()
        mock_model.from_pretrained.return_value.to.return_value = mock_model_instance

        # Mock the model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 197, 768)
        mock_model_instance.return_value = mock_output

        # Mock image loading
        mock_pil_image = MagicMock()
        mock_image.open.return_value.convert.return_value = mock_pil_image

        embedder = DINOEmbedder()
        embedder.embed(Path("/fake/path/image.png"))

        # Verify the processor was called with the image
        mock_processor_instance.assert_called()
