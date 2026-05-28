"""Tests for ContextStore — uses only built-in HashEmbedder + InMemoryBackend."""
import time
import pytest
from context_store import ContextStore, ContextEntry, SearchResult
from context_store.backends.memory import InMemoryBackend
from context_store.backends.sqlite import SQLiteBackend
from context_store.embeddings.hash import HashEmbedder


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    return ContextStore()   # in-memory + hash embedder


@pytest.fixture
def sqlite_store(tmp_path):
    db = str(tmp_path / "test.db")
    return ContextStore(backend=SQLiteBackend(db), embedder=HashEmbedder())


# ── Basic CRUD ────────────────────────────────────────────────────────────────

class TestAdd:
    def test_add_returns_entry(self, store):
        entry = store.add("hello world")
        assert isinstance(entry, ContextEntry)
        assert entry.text == "hello world"
        assert len(entry.embedding) > 0
        assert entry.id

    def test_add_with_metadata(self, store):
        entry = store.add("test", metadata={"source": "chat", "user": "alice"})
        assert entry.metadata["source"] == "chat"
        assert entry.metadata["user"] == "alice"

    def test_add_with_namespace(self, store):
        store.add("alice context", namespace="user:alice")
        store.add("bob context", namespace="user:bob")
        assert store.count("user:alice") == 1
        assert store.count("user:bob") == 1

    def test_add_many(self, store):
        texts = ["one", "two", "three"]
        entries = store.add_many(texts)
        assert len(entries) == 3
        assert store.count() == 3

    def test_add_many_with_metadata(self, store):
        texts = ["a", "b"]
        metas = [{"idx": 0}, {"idx": 1}]
        entries = store.add_many(texts, metadatas=metas)
        assert entries[0].metadata["idx"] == 0
        assert entries[1].metadata["idx"] == 1

    def test_custom_id(self, store):
        entry = store.add("custom", entry_id="my-custom-id")
        assert entry.id == "my-custom-id"
        assert store.get("my-custom-id") is not None


class TestGet:
    def test_get_existing(self, store):
        entry = store.add("retrieve me")
        fetched = store.get(entry.id)
        assert fetched is not None
        assert fetched.text == "retrieve me"

    def test_get_missing_returns_none(self, store):
        assert store.get("nonexistent-id") is None


class TestDelete:
    def test_delete_existing(self, store):
        entry = store.add("to be deleted")
        assert store.delete(entry.id) is True
        assert store.get(entry.id) is None

    def test_delete_missing_returns_false(self, store):
        assert store.delete("ghost") is False

    def test_delete_reduces_count(self, store):
        entry = store.add("x")
        assert store.count() == 1
        store.delete(entry.id)
        assert store.count() == 0


class TestList:
    def test_list_returns_all(self, store):
        store.add("a")
        store.add("b")
        store.add("c")
        assert len(store.list()) == 3

    def test_list_namespace_isolation(self, store):
        store.add("alice", namespace="ns:a")
        store.add("bob",   namespace="ns:b")
        assert len(store.list("ns:a")) == 1
        assert len(store.list("ns:b")) == 1

    def test_clear_namespace(self, store):
        store.add("a", namespace="ns:x")
        store.add("b", namespace="ns:x")
        store.add("c", namespace="ns:y")
        n = store.clear("ns:x")
        assert n == 2
        assert store.count("ns:x") == 0
        assert store.count("ns:y") == 1


# ── Search ────────────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_returns_results(self, store):
        store.add("payment info: Visa 4242")
        store.add("preferred theme: dark mode")
        store.add("location: San Francisco")
        results = store.search("credit card", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_respects_top_k(self, store):
        for i in range(10):
            store.add(f"entry {i}")
        results = store.search("entry", top_k=3)
        assert len(results) <= 3

    def test_search_namespace_isolation(self, store):
        store.add("alice info", namespace="alice")
        store.add("bob info",   namespace="bob")
        results = store.search("info", namespace="alice")
        assert all(r.entry.namespace == "alice" for r in results)

    def test_search_min_score(self, store):
        store.add("some text")
        results = store.search("some text", min_score=0.99)
        # HashEmbedder is deterministic so same text = score 1.0
        assert len(results) == 1
        assert results[0].score >= 0.99

    def test_search_filter_metadata(self, store):
        store.add("from chat",    metadata={"source": "chat"})
        store.add("from email",   metadata={"source": "email"})
        results = store.search("message", filter_metadata={"source": "chat"})
        assert all(r.entry.metadata["source"] == "chat" for r in results)

    def test_search_empty_store(self, store):
        results = store.search("anything")
        assert results == []

    def test_scores_sorted_descending(self, store):
        for i in range(5):
            store.add(f"item {i}")
        results = store.search("item")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ── TTL / Expiry ──────────────────────────────────────────────────────────────

class TestTTL:
    def test_expired_entry_not_returned(self, store):
        entry = store.add("temporary", ttl=0.01)
        assert store.get(entry.id) is not None
        time.sleep(0.05)
        assert store.get(entry.id) is None

    def test_expired_entry_excluded_from_list(self, store):
        store.add("persistent")
        store.add("ephemeral", ttl=0.01)
        time.sleep(0.05)
        entries = store.list()
        assert len(entries) == 1
        assert entries[0].text == "persistent"

    def test_expired_entry_excluded_from_search(self, store):
        store.add("searchable", ttl=0.01)
        time.sleep(0.05)
        results = store.search("searchable")
        assert results == []

    def test_default_ttl(self):
        store = ContextStore(default_ttl=0.01)
        store.add("with default ttl")
        time.sleep(0.05)
        assert store.count() == 0


# ── Context block ─────────────────────────────────────────────────────────────

class TestContextBlock:
    def test_returns_string(self, store):
        store.add("user likes Python")
        block = store.build_context_block("programming language")
        assert isinstance(block, str)
        assert len(block) > 0

    def test_empty_store_returns_empty_string(self, store):
        block = store.build_context_block("anything")
        assert block == ""

    def test_header_included(self, store):
        store.add("some context")
        block = store.build_context_block("context", header="MEMORY:")
        assert "MEMORY:" in block


# ── SQLite backend ────────────────────────────────────────────────────────────

class TestSQLiteBackend:
    def test_persistence(self, tmp_path):
        db = str(tmp_path / "persist.db")
        store1 = ContextStore(backend=SQLiteBackend(db), embedder=HashEmbedder())
        store1.add("persisted entry")

        store2 = ContextStore(backend=SQLiteBackend(db), embedder=HashEmbedder())
        assert store2.count() == 1
        assert store2.list()[0].text == "persisted entry"

    def test_sqlite_search(self, sqlite_store):
        sqlite_store.add("stored in sqlite")
        results = sqlite_store.search("stored")
        assert len(results) > 0

    def test_sqlite_clear(self, sqlite_store):
        sqlite_store.add("a")
        sqlite_store.add("b")
        n = sqlite_store.clear()
        assert n == 2
        assert sqlite_store.count() == 0


# ── Convenience constructors ──────────────────────────────────────────────────

class TestConstructors:
    def test_in_memory(self):
        store = ContextStore.in_memory()
        store.add("hello")
        assert store.count() == 1

    def test_from_sqlite(self, tmp_path):
        db = str(tmp_path / "cs.db")
        store = ContextStore.from_sqlite(db)
        store.add("hello sqlite")
        assert store.count() == 1
