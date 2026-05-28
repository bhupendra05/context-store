"""Abstract backend interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import ContextEntry


class BaseBackend(ABC):
    """Storage backend interface for ContextStore."""

    @abstractmethod
    def save(self, entry: ContextEntry) -> None:
        """Persist a ContextEntry."""
        ...

    @abstractmethod
    def get(self, entry_id: str) -> Optional[ContextEntry]:
        """Retrieve an entry by ID. Returns None if not found or expired."""
        ...

    @abstractmethod
    def delete(self, entry_id: str) -> bool:
        """Delete an entry. Returns True if it existed."""
        ...

    @abstractmethod
    def list(self, namespace: str = "default") -> List[ContextEntry]:
        """Return all non-expired entries in a namespace."""
        ...

    @abstractmethod
    def get_all_embeddings(self, namespace: str = "default") -> List[ContextEntry]:
        """Return all non-expired entries with embeddings for search."""
        ...

    @abstractmethod
    def clear(self, namespace: str = "default") -> int:
        """Delete all entries in namespace. Returns count deleted."""
        ...

    @abstractmethod
    def count(self, namespace: str = "default") -> int:
        """Return count of non-expired entries in namespace."""
        ...
