"""Base embedder interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract base class for text embedders."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Override for batch efficiency."""
        return [self.embed(t) for t in texts]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...
