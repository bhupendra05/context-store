"""
Hash-based embedder — zero dependencies, useful for testing.

Not semantically meaningful, but lets you run the full pipeline
without installing sentence-transformers or calling an LLM API.
"""
from __future__ import annotations

import hashlib
import math
from typing import List

from .base import BaseEmbedder

_DIM = 128


class HashEmbedder(BaseEmbedder):
    """
    Deterministic pseudo-embedder based on SHA-256.

    Produces unit vectors from text hashes. Semantically useless
    (similar texts will NOT have similar embeddings), but allows
    full pipeline testing with zero external dependencies.
    """

    @property
    def dimension(self) -> int:
        return _DIM

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Extend to _DIM floats by cycling through the digest bytes
        values: List[float] = []
        for i in range(_DIM):
            byte_val = digest[i % len(digest)]
            # Map [0,255] → [-1.0, 1.0]
            values.append((byte_val / 127.5) - 1.0)
        # Normalise to unit vector
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]
