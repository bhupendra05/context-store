"""
context-store — semantic memory store for LLM applications.

Quick Start
-----------
    from context_store import ContextStore

    store = ContextStore()
    store.add("User prefers dark mode")
    store.add("Payment method is Visa ending 4242")

    results = store.search("what card does the user have?")
    print(results[0].text)
"""

from .store import ContextStore
from .models import ContextEntry, SearchResult
from .backends.memory import InMemoryBackend
from .backends.sqlite import SQLiteBackend
from .embeddings.hash import HashEmbedder
from .embeddings.sentence_transformers import SentenceTransformerEmbedder
from .embeddings.openai import OpenAIEmbedder

__version__ = "0.1.0"
__all__ = [
    "ContextStore",
    "ContextEntry",
    "SearchResult",
    "InMemoryBackend",
    "SQLiteBackend",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
]
