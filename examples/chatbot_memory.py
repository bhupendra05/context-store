"""
Chatbot memory with context-store + Claude API.

pip install anthropic context-store
export ANTHROPIC_API_KEY=...
"""
import os
from context_store import ContextStore
from context_store.backends.sqlite import SQLiteBackend
from context_store.embeddings.hash import HashEmbedder  # swap for SentenceTransformerEmbedder

try:
    import anthropic
    CLIENT = anthropic.Anthropic()
except ImportError:
    CLIENT = None

# Persistent memory store (survives restarts)
memory = ContextStore(
    backend=SQLiteBackend("chatbot_memory.db"),
    embedder=HashEmbedder(),
    default_ttl=60 * 60 * 24 * 7,   # 7-day TTL
)


def chat(user_message: str, session_id: str = "default") -> str:
    """
    Send a message to Claude with relevant memory injected.

    1. Retrieve relevant past context
    2. Build system prompt with context block
    3. Call Claude
    4. Store the user message as new context
    """
    # 1. Retrieve relevant memories
    context_block = memory.build_context_block(
        query=user_message,
        top_k=5,
        namespace=f"session:{session_id}",
        header="What I know about this user:",
    )

    # 2. Build system prompt
    system = "You are a helpful assistant with memory of past conversations."
    if context_block:
        system += f"\n\n{context_block}"

    # 3. Call Claude (or mock if no API key)
    if CLIENT and os.getenv("ANTHROPIC_API_KEY"):
        response = CLIENT.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_message}],
        )
        reply = response.content[0].text
    else:
        reply = f"[MOCK] Responding to: {user_message!r}\nSystem had context: {bool(context_block)}"

    # 4. Store new memory
    memory.add(
        f"User said: {user_message}",
        metadata={"type": "user_message"},
        namespace=f"session:{session_id}",
    )

    return reply


def remember(fact: str, session_id: str = "default") -> None:
    """Explicitly store a fact about the user."""
    memory.add(fact, metadata={"type": "explicit_fact"}, namespace=f"session:{session_id}")
    print(f"✓ Remembered: {fact}")


if __name__ == "__main__":
    sid = "demo-user-1"

    # Pre-load some facts
    remember("The user's name is Alice", sid)
    remember("Alice prefers concise answers", sid)
    remember("Alice is a Python developer", sid)

    # Simulate a conversation
    for msg in [
        "What programming language should I use for a new web API?",
        "Can you summarise what you know about me?",
    ]:
        print(f"\n👤 {msg}")
        reply = chat(msg, sid)
        print(f"🤖 {reply}")

    print(f"\n📦 Total memories: {memory.count(f'session:{sid}')}")
