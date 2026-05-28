"""
Sentence-Transformers embedder.

Install: pip install sentence-transformers
Default model: all-MiniLM-L6-v2 (384-dim, ~80 MB, runs on CPU)
"""
from __future__ import annotations

from typing import List, Optional

from .base import BaseEmbedder

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder backed by a local sentence-transformers model.

    Example
    -------
    >>> embedder = SentenceTransformerEmbedder()
    >>> vec = embedder.embed("The user prefers dark mode")
    >>> len(vec)
    384
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL, device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedder.\n"
                "Install it with: pip install sentence-transformers"
            ) from e

        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._dim: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dim is None:
            sample = self._model.encode(["ping"], convert_to_numpy=True)
            self._dim = int(sample.shape[1])
        return self._dim

    def embed(self, text: str) -> List[float]:
        vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return vec[0].tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return [v.tolist() for v in vecs]
