"""
Data models for context-store.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextEntry:
    """A single piece of context stored in the store."""

    text: str
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    namespace: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "namespace": self.namespace,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextEntry":
        return cls(**data)


@dataclass
class SearchResult:
    """A single search result with similarity score."""

    entry: ContextEntry
    score: float  # cosine similarity [0.0, 1.0]

    @property
    def text(self) -> str:
        return self.entry.text

    @property
    def metadata(self) -> Dict[str, Any]:
        return self.entry.metadata

    @property
    def id(self) -> str:
        return self.entry.id

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.3f}, text={self.text[:60]!r})"
