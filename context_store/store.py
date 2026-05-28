"""
ContextStore — the main public API.
"""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Union

from .backends.base import BaseBackend
from .backends.memory import InMemoryBackend
from .backends.sqlite import SQLiteBackend
from .embeddings.base import BaseEmbedder
from .embeddings.hash import HashEmbedder
from .models import ContextEntry, SearchResult


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Fast cosine similarity for pre-normalised vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Clamp to [-1, 1] to avoid floating-point errors with unit vectors
    return max(-1.0, min(1.0, dot))


class ContextStore:
    """
    Semantic memory store for LLM applications.

    Stores text entries with embeddings and retrieves the most
    semantically relevant ones for a given query — in microseconds
    for small stores, or milliseconds at scale.

    Quick Start
    -----------
    >>> store = ContextStore()                   # in-memory, hash embedder
    >>> store.add("The user prefers dark mode")
    >>> store.add("Payment method is Visa ending 4242")
    >>> store.add("User is based in San Francisco")
    >>> results = store.search("what payment card does the user have?", top_k=1)
    >>> print(results[0].text)
    Payment method is Visa ending 4242

    Production Setup
    ----------------
    >>> from context_store.backends.sqlite import SQLiteBackend
    >>> from context_store.embeddings.sentence_transformers import SentenceTransformerEmbedder
    >>> store = ContextStore(
    ...     backend=SQLiteBackend("memory.db"),
    ...     embedder=SentenceTransformerEmbedder(),
    ... )

    Namespaces (per-user / per-session isolation)
    ---------------------------------------------
    >>> store.add("User likes Python", namespace="user:alice")
    >>> store.add("User likes Go",     namespace="user:bob")
    >>> store.search("programming language", namespace="user:alice")
    # Only returns Alice's context
    """

    def __init__(
        self,
        backend: Optional[BaseBackend] = None,
        embedder: Optional[BaseEmbedder] = None,
        default_namespace: str = "default",
        default_ttl: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        backend:
            Storage backend. Defaults to InMemoryBackend.
            Use SQLiteBackend("file.db") for persistence.
        embedder:
            Text embedder. Defaults to HashEmbedder (no deps, not semantic).
            Use SentenceTransformerEmbedder() for real semantic search.
        default_namespace:
            Namespace applied when none is specified.
        default_ttl:
            Default time-to-live in seconds. None = entries never expire.
        """
        self._backend: BaseBackend = backend or InMemoryBackend()
        self._embedder: BaseEmbedder = embedder or HashEmbedder()
        self._default_namespace = default_namespace
        self._default_ttl = default_ttl

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
        entry_id: Optional[str] = None,
    ) -> ContextEntry:
        """
        Add a text entry to the store.

        Parameters
        ----------
        text:       The text to store and embed.
        metadata:   Arbitrary key-value pairs attached to this entry.
        namespace:  Scope this entry to a namespace (e.g. "user:123").
        ttl:        Time-to-live in seconds. Overrides default_ttl.
        entry_id:   Optional custom ID. Auto-generated if omitted.

        Returns
        -------
        The created ContextEntry.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (time.time() + effective_ttl) if effective_ttl else None

        entry = ContextEntry(
            text=text,
            metadata=metadata or {},
            namespace=namespace or self._default_namespace,
            expires_at=expires_at,
        )
        if entry_id:
            entry.id = entry_id

        entry.embedding = self._embedder.embed(text)
        self._backend.save(entry)
        return entry

    def add_many(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
        ttl: Optional[float] = None,
    ) -> List[ContextEntry]:
        """Add multiple entries efficiently (batch embedding)."""
        metadatas = metadatas or [{} for _ in texts]
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = time.time()
        expires_at = (now + effective_ttl) if effective_ttl else None
        ns = namespace or self._default_namespace

        embeddings = self._embedder.embed_many(texts)
        entries = []
        for text, meta, emb in zip(texts, metadatas, embeddings):
            entry = ContextEntry(
                text=text,
                embedding=emb,
                metadata=meta,
                namespace=ns,
                expires_at=expires_at,
            )
            self._backend.save(entry)
            entries.append(entry)
        return entries

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        min_score: float = 0.0,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Find the most semantically relevant entries for a query.

        Parameters
        ----------
        query:           Natural language query.
        top_k:           Maximum number of results to return.
        namespace:       Scope search to this namespace.
        min_score:       Minimum cosine similarity threshold [0.0 – 1.0].
        filter_metadata: Only return entries whose metadata contains all
                         these key-value pairs.

        Returns
        -------
        List of SearchResult objects sorted by descending similarity score.

        Example
        -------
        >>> results = store.search("payment info", top_k=3, min_score=0.5)
        >>> for r in results:
        ...     print(f"{r.score:.2f}  {r.text}")
        """
        ns = namespace or self._default_namespace
        candidates = self._backend.get_all_embeddings(ns)

        if not candidates:
            return []

        query_vec = self._embedder.embed(query)

        results: List[SearchResult] = []
        for entry in candidates:
            if filter_metadata:
                if not all(entry.metadata.get(k) == v for k, v in filter_metadata.items()):
                    continue
            score = _cosine_similarity(query_vec, entry.embedding)
            if score >= min_score:
                results.append(SearchResult(entry=entry, score=score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, entry_id: str) -> Optional[ContextEntry]:
        """Retrieve a specific entry by ID."""
        return self._backend.get(entry_id)

    def list(self, namespace: Optional[str] = None) -> List[ContextEntry]:
        """List all entries in a namespace (insertion order)."""
        return self._backend.list(namespace or self._default_namespace)

    def count(self, namespace: Optional[str] = None) -> int:
        """Count entries in a namespace."""
        return self._backend.count(namespace or self._default_namespace)

    # ------------------------------------------------------------------
    # Delete / Management
    # ------------------------------------------------------------------

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns True if it existed."""
        return self._backend.delete(entry_id)

    def clear(self, namespace: Optional[str] = None) -> int:
        """Delete all entries in a namespace. Returns count deleted."""
        return self._backend.clear(namespace or self._default_namespace)

    # ------------------------------------------------------------------
    # Context window helper
    # ------------------------------------------------------------------

    def build_context_block(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
        max_tokens: int = 2000,
        header: str = "Relevant context:",
        separator: str = "\n- ",
    ) -> str:
        """
        Build a ready-to-inject context block for LLM prompts.

        Retrieves the most relevant entries and formats them as a
        single string you can prepend to your system prompt.

        Example
        -------
        >>> block = store.build_context_block("What does the user prefer?")
        >>> messages = [
        ...     {"role": "system", "content": f"You are a helpful assistant.\\n\\n{block}"},
        ...     {"role": "user", "content": user_message},
        ... ]
        """
        results = self.search(query, top_k=top_k, namespace=namespace)
        if not results:
            return ""

        lines = [header]
        token_estimate = len(header) // 4
        for r in results:
            line = separator + r.text
            token_estimate += len(line) // 4
            if token_estimate > max_tokens:
                break
            lines.append(line)

        return "".join(lines)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_sqlite(
        cls,
        path: str = "context_store.db",
        embedder: Optional[BaseEmbedder] = None,
        **kwargs: Any,
    ) -> "ContextStore":
        """Create a store backed by SQLite."""
        return cls(backend=SQLiteBackend(path), embedder=embedder, **kwargs)

    @classmethod
    def in_memory(
        cls,
        embedder: Optional[BaseEmbedder] = None,
        **kwargs: Any,
    ) -> "ContextStore":
        """Create an ephemeral in-memory store."""
        return cls(backend=InMemoryBackend(), embedder=embedder, **kwargs)
