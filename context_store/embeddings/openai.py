"""
OpenAI embedder — uses text-embedding-3-small by default.

Install: pip install openai
"""
from __future__ import annotations

import os
from typing import List, Optional

from .base import BaseEmbedder

_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIM = 1536


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder backed by the OpenAI Embeddings API.

    Example
    -------
    >>> embedder = OpenAIEmbedder()   # reads OPENAI_API_KEY from env
    >>> vec = embedder.embed("Hello world")
    >>> len(vec)
    1536
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai is required for OpenAIEmbedder.\n"
                "Install it with: pip install openai"
            ) from e

        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def dimension(self) -> int:
        return self._dimensions or _DEFAULT_DIM

    def embed(self, text: str) -> List[float]:
        kwargs: dict = {"model": self._model, "input": [text]}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return response.data[0].embedding

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        kwargs: dict = {"model": self._model, "input": texts}
        if self._dimensions:
            kwargs["dimensions"] = self._dimensions
        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]
