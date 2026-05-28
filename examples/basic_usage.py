"""
Basic usage of context-store.

Runs with zero dependencies (uses built-in HashEmbedder).
For real semantic search: pip install sentence-transformers
"""
from context_store import ContextStore

# ── 1. In-memory store (zero config) ─────────────────────────────────────────
store = ContextStore()

store.add("The user's name is Alice")
store.add("Alice prefers dark mode in all apps")
store.add("Alice's payment method is Visa ending in 4242")
store.add("Alice is based in San Francisco, California")
store.add("Alice's preferred programming language is Python")
store.add("Alice's last order was a standing desk, delivered on May 10")

# ── 2. Search ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("SEARCH: 'what credit card does the user have?'")
print("=" * 60)
results = store.search("what credit card does the user have?", top_k=3)
for r in results:
    print(f"  [{r.score:.3f}] {r.text}")

# ── 3. Build a context block for LLM prompts ──────────────────────────────────
print("\n" + "=" * 60)
print("CONTEXT BLOCK for 'user location and name'")
print("=" * 60)
block = store.build_context_block("user location and name", top_k=3)
print(block)

# ── 4. Namespaces ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NAMESPACES — per-user isolation")
print("=" * 60)
store.add("Bob likes Go",   namespace="user:bob")
store.add("Bob lives in NYC", namespace="user:bob")

alice_results = store.search("where does the user live?", namespace="user:alice")
bob_results   = store.search("where does the user live?", namespace="user:bob")
print(f"Alice results: {[r.text for r in alice_results[:2]]}")
print(f"Bob results:   {[r.text for r in bob_results[:2]]}")

# ── 5. TTL / expiry ───────────────────────────────────────────────────────────
import time
print("\n" + "=" * 60)
print("TTL — expiring context")
print("=" * 60)
entry = store.add("Temporary session token: abc123", ttl=1.0)
print(f"Before expiry: {store.get(entry.id)}")
time.sleep(1.1)
print(f"After expiry:  {store.get(entry.id)}")   # None

print("\n✅ Done!")
