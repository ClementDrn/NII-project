from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    @abstractmethod
    def embed(self, image_path: Path) -> torch.Tensor:
        """
        Generate embedding for the given image.
        """
        pass


class DINOEmbedder(Embedder):
    def __init__(self):
        super().__init__()
        # device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self._device = torch.device('cpu')
        self._processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')    # 86 million parameters, size of 331 MB
        self._model = AutoModel.from_pretrained('facebook/dinov2-base').to(self._device)

    def embed(self, image_path: Path) -> torch.Tensor:
        """
        Generate DINO embedding for the given image.
        """
        # Proccess image and embed it into a feature vector
        image = Image.open(image_path).convert("RGB")
        image_features = None
        with torch.no_grad():
            # Process image
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            outputs = self._model(**inputs)

            # Extract features
            image_features = outputs.last_hidden_state  # [B, N, D]
            # Mean pooling over tokens, better for image similarity
            image_features = image_features.mean(dim=1)     # [B, D]
            # Normalize features
            image_features = torch.nn.functional.normalize(image_features, dim=-1)
            # Take only one image in the batch
            image_features = image_features[0]              # [D]

        return image_features.cpu()
