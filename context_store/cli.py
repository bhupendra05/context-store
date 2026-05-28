"""
context-store CLI

Usage:
    context-store add  "User prefers dark mode" --db memory.db
    context-store search "what theme does the user like?" --db memory.db
    context-store list  --db memory.db
    context-store clear --db memory.db
"""
from __future__ import annotations

import json
import sys

import click

from .backends.sqlite import SQLiteBackend
from .embeddings.hash import HashEmbedder
from .store import ContextStore

_DEFAULT_DB = "context_store.db"


def _make_store(db: str) -> ContextStore:
    return ContextStore(backend=SQLiteBackend(db), embedder=HashEmbedder())


@click.group()
@click.version_option()
def cli() -> None:
    """context-store — semantic memory store for LLM applications."""


@cli.command()
@click.argument("text")
@click.option("--db", default=_DEFAULT_DB, show_default=True, help="SQLite database path")
@click.option("--namespace", "-n", default="default", show_default=True)
@click.option("--ttl", type=float, default=None, help="Time-to-live in seconds")
@click.option("--meta", type=str, default=None, help='JSON metadata, e.g. \'{"source":"chat"}\'')
def add(text: str, db: str, namespace: str, ttl: float | None, meta: str | None) -> None:
    """Add a context entry to the store."""
    metadata = json.loads(meta) if meta else {}
    store = _make_store(db)
    entry = store.add(text, metadata=metadata, namespace=namespace, ttl=ttl)
    click.echo(click.style("✓ Added", fg="green") + f"  [{entry.id[:8]}]  {text[:80]}")


@cli.command()
@click.argument("query")
@click.option("--db", default=_DEFAULT_DB, show_default=True)
@click.option("--namespace", "-n", default="default", show_default=True)
@click.option("--top-k", "-k", default=5, show_default=True)
@click.option("--min-score", default=0.0, show_default=True)
def search(query: str, db: str, namespace: str, top_k: int, min_score: float) -> None:
    """Search for semantically relevant context entries."""
    store = _make_store(db)
    results = store.search(query, top_k=top_k, namespace=namespace, min_score=min_score)
    if not results:
        click.echo(click.style("No results found.", fg="yellow"))
        return
    click.echo(f"\n🔍  Results for: {click.style(query, bold=True)}\n")
    for i, r in enumerate(results, 1):
        score_color = "green" if r.score > 0.8 else "yellow" if r.score > 0.5 else "red"
        score_str = click.style(f"{r.score:.3f}", fg=score_color)
        click.echo(f"  {i}. [{score_str}]  {r.text}")
        if r.metadata:
            click.echo(f"       {click.style('meta:', dim=True)} {r.metadata}")
    click.echo()


@cli.command(name="list")
@click.option("--db", default=_DEFAULT_DB, show_default=True)
@click.option("--namespace", "-n", default="default", show_default=True)
def list_cmd(db: str, namespace: str) -> None:
    """List all entries in a namespace."""
    store = _make_store(db)
    entries = store.list(namespace=namespace)
    if not entries:
        click.echo(click.style("No entries found.", fg="yellow"))
        return
    click.echo(f"\n📋  {len(entries)} entries in namespace '{namespace}':\n")
    for e in entries:
        import datetime
        ts = datetime.datetime.fromtimestamp(e.created_at).strftime("%Y-%m-%d %H:%M")
        expires = ""
        if e.expires_at:
            exp_ts = datetime.datetime.fromtimestamp(e.expires_at).strftime("%H:%M")
            expires = click.style(f"  expires {exp_ts}", fg="yellow")
        click.echo(f"  [{e.id[:8]}] {ts}{expires}  {e.text[:80]}")
    click.echo()


@cli.command()
@click.argument("entry_id")
@click.option("--db", default=_DEFAULT_DB, show_default=True)
def delete(entry_id: str, db: str) -> None:
    """Delete an entry by ID (or ID prefix)."""
    store = _make_store(db)
    # Support prefix matching
    if len(entry_id) < 36:
        all_entries = store.list()
        matches = [e for e in all_entries if e.id.startswith(entry_id)]
        if not matches:
            click.echo(click.style(f"No entry found with ID prefix '{entry_id}'", fg="red"))
            sys.exit(1)
        if len(matches) > 1:
            click.echo(click.style(f"Multiple matches for '{entry_id}'. Be more specific.", fg="red"))
            sys.exit(1)
        entry_id = matches[0].id

    ok = store.delete(entry_id)
    if ok:
        click.echo(click.style("✓ Deleted", fg="green") + f"  {entry_id[:8]}")
    else:
        click.echo(click.style("Not found", fg="yellow"))


@cli.command()
@click.option("--db", default=_DEFAULT_DB, show_default=True)
@click.option("--namespace", "-n", default="default", show_default=True)
@click.confirmation_option(prompt="This will delete all entries. Continue?")
def clear(db: str, namespace: str) -> None:
    """Delete all entries in a namespace."""
    store = _make_store(db)
    n = store.clear(namespace=namespace)
    click.echo(click.style(f"✓ Cleared {n} entries", fg="green") + f" from namespace '{namespace}'")


@cli.command()
@click.option("--db", default=_DEFAULT_DB, show_default=True)
@click.option("--namespace", "-n", default="default", show_default=True)
def stats(db: str, namespace: str) -> None:
    """Show store statistics."""
    store = _make_store(db)
    n = store.count(namespace=namespace)
    click.echo(f"\n📊  context-store stats\n")
    click.echo(f"   Database  : {db}")
    click.echo(f"   Namespace : {namespace}")
    click.echo(f"   Entries   : {n}")
    click.echo()


if __name__ == "__main__":
    cli()
