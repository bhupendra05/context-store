"""SQLite backend — persistent, zero extra dependencies."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import List, Optional

from .base import BaseBackend
from ..models import ContextEntry

_DDL = """
CREATE TABLE IF NOT EXISTS context_entries (
    id          TEXT PRIMARY KEY,
    namespace   TEXT NOT NULL DEFAULT 'default',
    text        TEXT NOT NULL,
    embedding   TEXT NOT NULL DEFAULT '[]',
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  REAL NOT NULL,
    expires_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_namespace ON context_entries (namespace);
CREATE INDEX IF NOT EXISTS idx_expires   ON context_entries (expires_at);
"""


class SQLiteBackend(BaseBackend):
    """
    Persistent SQLite-backed storage.

    Perfect for:
    - Single-machine deployments
    - Long-lived chatbot sessions
    - Development with durability

    Example
    -------
    >>> backend = SQLiteBackend("context.db")
    >>> backend = SQLiteBackend(":memory:")   # in-memory SQLite
    """

    def __init__(self, path: str = "context_store.db") -> None:
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    def _row_to_entry(self, row: sqlite3.Row) -> ContextEntry:
        return ContextEntry(
            id=row["id"],
            namespace=row["namespace"],
            text=row["text"],
            embedding=json.loads(row["embedding"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            expires_at=row["expires_at"],
        )

    def save(self, entry: ContextEntry) -> None:
        self._conn.execute(
            """INSERT OR REPLACE INTO context_entries
               (id, namespace, text, embedding, metadata, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.id,
                entry.namespace,
                entry.text,
                json.dumps(entry.embedding),
                json.dumps(entry.metadata),
                entry.created_at,
                entry.expires_at,
            ),
        )
        self._conn.commit()

    def get(self, entry_id: str) -> Optional[ContextEntry]:
        row = self._conn.execute(
            "SELECT * FROM context_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        entry = self._row_to_entry(row)
        if entry.is_expired:
            self.delete(entry_id)
            return None
        return entry

    def delete(self, entry_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM context_entries WHERE id = ?", (entry_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def _purge_expired(self) -> None:
        self._conn.execute(
            "DELETE FROM context_entries WHERE expires_at IS NOT NULL AND expires_at < ?",
            (time.time(),),
        )
        self._conn.commit()

    def list(self, namespace: str = "default") -> List[ContextEntry]:
        self._purge_expired()
        rows = self._conn.execute(
            "SELECT * FROM context_entries WHERE namespace = ? ORDER BY created_at",
            (namespace,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_all_embeddings(self, namespace: str = "default") -> List[ContextEntry]:
        return [e for e in self.list(namespace) if e.embedding]

    def clear(self, namespace: str = "default") -> int:
        cur = self._conn.execute(
            "DELETE FROM context_entries WHERE namespace = ?", (namespace,)
        )
        self._conn.commit()
        return cur.rowcount

    def count(self, namespace: str = "default") -> int:
        self._purge_expired()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM context_entries WHERE namespace = ?", (namespace,)
        ).fetchone()
        return row[0]

    def close(self) -> None:
        self._conn.close()
