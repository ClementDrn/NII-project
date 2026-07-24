from untitled.generator.batch import Batch, BatchRequest

from abc import ABC, abstractmethod
from typing import Optional


class ProviderAdapter(ABC):
    """
    Abstract Base Class defining the standard interface for all model providers.
    """
    
    @abstractmethod
    async def open(self) -> None:
        """Initialize the client or establish connections if required."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources."""
        ...

    @abstractmethod
    async def test_connection(self) -> bool:
        """Execute a minimal prompt to verify provider connectivity."""
        ...

    @abstractmethod
    async def send_prompt(self, system_text: str = "", user_text: str = "", user_images: list[bytes] = []) -> str:
        """
        Send a standardized OpenAI-formatted message payload to the provider.
        
        Args:
            system_text: The system prompt text
            user_text: The user prompt text
            user_images: List of user-provided images in bytes format
            
        Returns:
            The extracted text response.
        """
        ...

    @abstractmethod
    async def create_batch(self, requests: list[BatchRequest]) -> Optional[Batch]:
        """
        Create a batch for processing multiple requests.
        
        Args:
            requests: List of BatchRequest objects
        
        Returns:
            A Batch object representing the created batch, or None if creation failed.
        """
