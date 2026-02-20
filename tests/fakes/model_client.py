"""
Fake ModelClient for testing.

Provides a duck-typed replacement for the real ModelClient that:
- Returns preset responses instead of calling actual LLM/VLM APIs
- Tracks all prompt calls for test verification
- Requires no model server or API keys
"""
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Any
from enum import Enum


class ModelType(Enum):
    """Type of model: VLM (Vision-Language Model) or LLM (Language Model)."""
    VLM = "vlm"
    LLM = "llm"


class FakeModelClient:
    """
    A fake ModelClient that returns preset responses without calling actual APIs.
    
    Usage:
        fake = FakeModelClient()
        fake.preset_responses = ["Response 1", "Response 2"]
        
        # Use fake in place of real ModelClient
        response = await fake.prompt("system", "user", [])
        
        # Verify calls
        assert len(fake.prompt_calls) == 1
        assert fake.prompt_calls[0]["user_text"] == "user"
    
    Alternative - use a callback for dynamic responses:
        fake.response_callback = lambda sys, user, imgs: f"Echo: {user}"
    """
    
    class State(Enum):
        DISABLED = 0
        STARTED = 1
        CONNECTED = 2
    
    def __init__(
        self,
        model_type: ModelType = ModelType.VLM,
        priority: int = 0,
        batch_size: int = 1,
        host: str = "http://fake-host:8000",
        api_key: str = "fake-api-key",
        api_parameters: Optional[dict] = None,
    ):
        """
        Initialize the fake model client.
        
        Args:
            model_type: Type of model (VLM or LLM).
            priority: Priority for prompt dispatcher.
            batch_size: Batch size hint.
            host: Fake host URL.
            api_key: Fake API key.
            api_parameters: Additional API parameters.
        """
        self.model_type = model_type
        self.priority = priority
        self.batch_size = batch_size
        self.host = host
        self.api_key = api_key
        self.api_parameters = api_parameters or {}
        
        # State management
        self.state = FakeModelClient.State.DISABLED
        self.is_hot = False
        self.client = None  # Mimics real client attribute
        
        # Preset responses (cycles through list)
        self.preset_responses: List[str] = ["fake model response"]
        self._response_index: int = 0
        
        # Alternative: callback for dynamic responses
        self.response_callback: Optional[Callable[[str, str, List[Path]], str]] = None
        
        # Call tracking for verification
        self.prompt_calls: List[dict] = []
        self.start_calls: int = 0
        self.stop_calls: int = 0
        self.open_connection_calls: int = 0
        self.close_connection_calls: int = 0
    
    async def prompt(
        self,
        system_text: str,
        user_text: str,
        user_image_paths: List[Path],
    ) -> str:
        """
        Fake prompting the model. Returns preset response and tracks the call.
        
        Args:
            system_text: System message.
            user_text: User message text.
            user_image_paths: List of image paths (recorded but not processed).
            
        Returns:
            Next preset response or callback result.
        """
        self.prompt_calls.append({
            "system_text": system_text,
            "user_text": user_text,
            "user_image_paths": list(user_image_paths),
        })
        
        # Use callback if provided
        if self.response_callback is not None:
            return self.response_callback(system_text, user_text, user_image_paths)
        
        # Cycle through preset responses
        if not self.preset_responses:
            return ""
        
        response = self.preset_responses[self._response_index]
        self._response_index = (self._response_index + 1) % len(self.preset_responses)
        return response
    
    async def start(self) -> None:
        """Start the model client (tracks call and updates state)."""
        self.start_calls += 1
        self.state = FakeModelClient.State.STARTED
    
    async def stop(self) -> None:
        """Stop the model client (tracks call and updates state)."""
        self.stop_calls += 1
        self.state = FakeModelClient.State.DISABLED
    
    async def set_hot(self) -> None:
        """Set the model client in a hot state."""
        self.is_hot = True
    
    async def open_connection(self) -> None:
        """Open connection (tracks call and updates state to CONNECTED)."""
        self.open_connection_calls += 1
        if self.state == FakeModelClient.State.DISABLED:
            await self.start()
        self.state = FakeModelClient.State.CONNECTED
    
    async def close_connection(self) -> None:
        """Close connection (tracks call)."""
        self.close_connection_calls += 1
        if self.state == FakeModelClient.State.CONNECTED:
            self.state = FakeModelClient.State.STARTED
    
    async def test_connection(self) -> bool:
        """Test connection (always returns True for fake)."""
        return True
    
    def is_remote(self) -> bool:
        """Return whether this is a remote model (always False for fake)."""
        return False
    
    def is_parallelizable(self) -> bool:
        """Return whether this model supports parallel requests."""
        return True
    
    def get_state(self) -> "FakeModelClient.State":
        """Get the current state."""
        return self.state
    
    def reset_tracking(self) -> None:
        """Reset all call tracking (useful between test cases)."""
        self.prompt_calls.clear()
        self.start_calls = 0
        self.stop_calls = 0
        self.open_connection_calls = 0
        self.close_connection_calls = 0
        self._response_index = 0
