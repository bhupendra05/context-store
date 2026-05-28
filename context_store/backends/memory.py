"""In-memory backend — fast, zero dependencies, not persistent."""
from __future__ import annotations

import time
from typing import Dict, List, Optional

from .base import BaseBackend
from ..models import ContextEntry


class InMemoryBackend(BaseBackend):
    """
    Thread-safe in-memory storage.

    Perfect for:
    - Single-session chatbots
    - Unit tests
    - Development / prototyping

    Data is lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: Dict[str, ContextEntry] = {}

    def save(self, entry: ContextEntry) -> None:
        self._store[entry.id] = entry

    def get(self, entry_id: str) -> Optional[ContextEntry]:
        entry = self._store.get(entry_id)
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[entry_id]
            return None
        return entry

    def delete(self, entry_id: str) -> bool:
        return self._store.pop(entry_id, None) is not None

    def list(self, namespace: str = "default") -> List[ContextEntry]:
        now = time.time()
        results = []
        expired = []
        for entry_id, entry in self._store.items():
            if entry.namespace != namespace:
                continue
            if entry.expires_at and now > entry.expires_at:
                expired.append(entry_id)
            else:
                results.append(entry)
        for eid in expired:
            del self._store[eid]
        return sorted(results, key=lambda e: e.created_at)

    def get_all_embeddings(self, namespace: str = "default") -> List[ContextEntry]:
        return [e for e in self.list(namespace) if e.embedding]

    def clear(self, namespace: str = "default") -> int:
        to_delete = [eid for eid, e in self._store.items() if e.namespace == namespace]
        for eid in to_delete:
            del self._store[eid]
        return len(to_delete)

    def count(self, namespace: str = "default") -> int:
        return len(self.list(namespace))
