# context-store

**Semantic memory store for LLM applications.**  
Store text, search by meaning, inject relevant context into prompts — in one import.

```bash
pip install context-store
```

[![CI](https://github.com/bhupendra05/context-store/actions/workflows/ci.yml/badge.svg)](https://github.com/bhupendra05/context-store/actions)
[![PyPI](https://img.shields.io/pypi/v/context-store)](https://pypi.org/project/context-store/)
[![Python](https://img.shields.io/pypi/pyversions/context-store)](https://pypi.org/project/context-store/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The problem

Every production LLM app needs to answer: *"What does the model need to know right now?"*

Stuffing the entire conversation history into every prompt:
- Blows through the context window
- Wastes tokens (and money)
- Buries the relevant signal in noise

`context-store` solves this by storing facts as embeddings and retrieving only the most semantically relevant ones for each query.

---

## Quick start

```python
from context_store import ContextStore

store = ContextStore()   # in-memory, zero dependencies

store.add("The user's name is Alice")
store.add("Alice prefers dark mode in all apps")
store.add("Alice's payment method is Visa ending in 4242")
store.add("Alice is based in San Francisco")
store.add("Alice's last order: standing desk, delivered May 10")

# Retrieve by meaning, not keyword
results = store.search("what payment card does the user have?", top_k=2)
for r in results:
    print(f"[{r.score:.3f}] {r.text}")
# [1.000] Alice's payment method is Visa ending in 4242
# [0.821] The user's name is Alice
```

---

## Inject context into LLM prompts

```python
# Build a ready-to-use context block
block = store.build_context_block(
    query="where does the user live?",
    top_k=3,
    header="What I know about this user:",
)

messages = [
    {"role": "system", "content": f"You are a helpful assistant.\n\n{block}"},
    {"role": "user",   "content": "Which timezone should I use for the meeting?"},
]
```

---

## Persistent storage (SQLite)

```python
from context_store import ContextStore
from context_store.backends.sqlite import SQLiteBackend
from context_store.embeddings.sentence_transformers import SentenceTransformerEmbedder

store = ContextStore(
    backend=SQLiteBackend("memory.db"),          # survives restarts
    embedder=SentenceTransformerEmbedder(),       # real semantic search
)

store.add("User loves Python")
# Restart your process — data persists
```

---

## Embedders

| Embedder | Install | Semantic? | Notes |
|---|---|---|---|
| `HashEmbedder` | _built-in_ | ❌ | Zero deps. Good for testing. |
| `SentenceTransformerEmbedder` | `pip install sentence-transformers` | ✅ | Local, free, ~80 MB |
| `OpenAIEmbedder` | `pip install openai` | ✅ | Best quality, API cost |

```python
# Switch embedder in one line
from context_store.embeddings.sentence_transformers import SentenceTransformerEmbedder
store = ContextStore(embedder=SentenceTransformerEmbedder())   # all-MiniLM-L6-v2

from context_store.embeddings.openai import OpenAIEmbedder
store = ContextStore(embedder=OpenAIEmbedder())   # text-embedding-3-small
```

---

## Namespaces — per-user isolation

```python
store.add("User likes Python", namespace="user:alice")
store.add("User likes Go",     namespace="user:bob")

# Each user only sees their own context
alice_ctx = store.search("language", namespace="user:alice")
bob_ctx   = store.search("language", namespace="user:bob")
```

---

## TTL — expiring context

```python
# Store a session token that auto-expires in 1 hour
store.add("Session token: abc123", ttl=3600)

# Set a default TTL for all entries
store = ContextStore(default_ttl=60 * 60 * 24)   # 24 hours
```

---

## Metadata filtering

```python
store.add("Message from support chat", metadata={"source": "chat", "priority": "high"})
store.add("Email from last week",      metadata={"source": "email", "priority": "low"})

# Only search high-priority chat messages
results = store.search("user issue", filter_metadata={"source": "chat", "priority": "high"})
```

---

## Batch insert

```python
texts = ["fact one", "fact two", "fact three"]
metas = [{"type": "fact"}] * 3

entries = store.add_many(texts, metadatas=metas)
# Single embedding call — much faster than looping add()
```

---

## CLI

```bash
# Add entries
context-store add "user prefers dark mode" --db memory.db
context-store add "payment: Visa 4242"    --db memory.db --meta '{"source":"checkout"}'

# Search
context-store search "what card does the user have?" --db memory.db --top-k 3

# List / manage
context-store list   --db memory.db
context-store stats  --db memory.db
context-store delete <id-prefix> --db memory.db
context-store clear  --db memory.db
```

---

## Backends

| Backend | Persistent | Dependencies | Best for |
|---|---|---|---|
| `InMemoryBackend` | ❌ | None | Testing, single-session bots |
| `SQLiteBackend` | ✅ | None | Single-machine production |

```python
# Convenience constructors
store = ContextStore.in_memory()
store = ContextStore.from_sqlite("app_memory.db")
```

---

## Architecture

```
ContextStore
├── Embedder          → text → float[]
│   ├── HashEmbedder          (zero deps, for testing)
│   ├── SentenceTransformerEmbedder  (local, semantic)
│   └── OpenAIEmbedder        (API, best quality)
└── Backend           → storage + retrieval
    ├── InMemoryBackend       (fast, ephemeral)
    └── SQLiteBackend         (persistent, zero deps)
```

Cosine similarity search runs in-process against all stored embeddings. For datasets > 100k entries, swap in a FAISS or pgvector backend (PRs welcome).

---

## Real-world example: chatbot memory

```python
from context_store import ContextStore
from context_store.backends.sqlite import SQLiteBackend
import anthropic

memory = ContextStore(backend=SQLiteBackend("bot_memory.db"))
client = anthropic.Anthropic()

def chat(user_message: str, user_id: str) -> str:
    # 1. Retrieve relevant memories
    context = memory.build_context_block(
        query=user_message,
        namespace=f"user:{user_id}",
        top_k=5,
    )
    # 2. Inject into system prompt
    system = f"You are a helpful assistant.\n\n{context}" if context else "You are a helpful assistant."
    # 3. Call Claude
    response = client.messages.create(
        model="claude-opus-4-5", max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    reply = response.content[0].text
    # 4. Remember this turn
    memory.add(f"User said: {user_message}", namespace=f"user:{user_id}")
    return reply
```

---

## Running tests

```bash
git clone https://github.com/bhupendra05/context-store
cd context-store
pip install -e ".[dev]"
pytest -v
```

---

## License

MIT © [Bhupendra Tale](https://github.com/bhupendra05)
