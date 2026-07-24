from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Any, AsyncIterator, List


@dataclass
class BatchRequest:
    """Represents a single request in a batch operation."""

    request_id: str
    system_text: str
    user_text: str | None = None
    user_images: list[bytes] | None = None

class BatchStatus(Enum):
    """Normalized batch status values used by the client API."""

    UNKNOWN = "unknown"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class BatchRequestStatus(Enum):
    """Normalized request status values used by the client API."""

    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    ERRORED = "errored"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

@dataclass
class BatchRequestResult:
    """Normalized result for a single request in a batch."""

    request_id: str
    status: BatchRequestStatus
    text: str
    thinking: str


# NOTE: The Batch implementation should be defined along the provider adapter.
class Batch(ABC):
    """
    Abstract base class representing a batch of requests.
    """
    
    def __init__(self, requests: List[BatchRequest]):
        self._batch_id: str = ""
        self._requests: List[BatchRequest] = requests
        self._status: BatchStatus = BatchStatus.UNKNOWN

    def get_id(self) -> str:
        """Return the unique identifier for the batch."""
        return self._batch_id

    def get_status(self) -> BatchStatus:
        """Return the current status of the batch."""
        return self._status

    @abstractmethod
    async def send(self) -> BatchStatus:
        """Send the batch to the model client."""
        ...

    @abstractmethod
    async def check_status(self) -> BatchStatus:
        """Check the status of the batch."""
        ...

    @abstractmethod
    def download_results(self) -> AsyncIterator[BatchRequestResult]:
        """Download the results of the batch if completed."""
        ...
