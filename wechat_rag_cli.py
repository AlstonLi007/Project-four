from __future__ import annotations

"""CLI helpers to tune per-contact RAG behavior.

Allows you to list contacts along with their current ``rag_mode`` and
``half_life_days`` settings or update them without opening a Python REPL.
"""

from typing import Optional

import typer

from db import Database, DEFAULT_DB_PATH

VALID_RAG_MODES = ("off", "light", "normal", "aggressive")

app = typer.Typer(help="Manage RAG tuning knobs for individual contacts.")


def _resolve_db(db_path: str) -> Database:
    return Database(path=db_path)


@app.command("list")
def list_contacts(
    db: str = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite DB."),
    limit: int = typer.Option(50, "--limit", "-n", help="Max contacts to display."),
) -> None:
    """Show contacts with their RAG mode and half-life settings."""

    store = _resolve_db(db)
    contacts = store.list_contacts()
    if not contacts:
        typer.echo("No contacts found.")
        raise typer.Exit(code=0)

    header = f"{'id':<6} {'wechat_id':<30} {'display_name':<30} {'rag_mode':<12} {'half_life_days':<14}"
    typer.echo(header)
    typer.echo("-" * len(header))

    for row in contacts[:limit]:
        cid = row.get("id")
        wechat_id = row.get("wechat_id") or ""
        display_name = row.get("display_name") or ""
        rag_mode = store.get_contact_rag_mode(cid, default="off")
        half_life = store.get_contact_half_life(cid, default=30.0)
        typer.echo(
            f"{str(cid):<6} {wechat_id[:30]:<30} {display_name[:30]:<30} {rag_mode:<12} {half_life:<14}"
        )


@app.command("set")
def set_contact_rag(
    wechat_id: str = typer.Argument(..., help="The contact's stable wechat_id."),
    rag_mode: Optional[str] = typer.Option(None, "--mode", "-m", help=str(VALID_RAG_MODES)),
    half_life_days: Optional[float] = typer.Option(None, "--half-life", "-h", help="Half-life in days."),
    display_name: Optional[str] = typer.Option(None, "--name", help="Optional display name when creating the contact."),
    db: str = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite DB."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show changes without writing."),
) -> None:
    """Update ``rag_mode`` and/or ``half_life_days`` for a contact."""

    if rag_mode is None and half_life_days is None:
        raise typer.BadParameter("Specify at least one of --mode or --half-life.")
    if rag_mode is not None and rag_mode not in VALID_RAG_MODES:
        raise typer.BadParameter(f"--mode must be one of: {', '.join(VALID_RAG_MODES)}")

    store = _resolve_db(db)
    contact = store.get_or_create_contact(wechat_id=wechat_id, display_name=display_name or wechat_id)

    before_mode = store.get_contact_rag_mode(contact.id, default="off")
    before_half_life = store.get_contact_half_life(contact.id, default=30.0)

    typer.echo(f"Updating contact {wechat_id} ({contact.display_name})")

    if rag_mode is not None:
        typer.echo(f"  rag_mode: {before_mode!r} -> {rag_mode!r}")
        if not dry_run:
            store.set_contact_rag_mode(contact.id, rag_mode)

    if half_life_days is not None:
        typer.echo(f"  half_life_days: {before_half_life!r} -> {half_life_days!r}")
        if not dry_run:
            store.set_contact_half_life(contact.id, half_life_days)

    if dry_run:
        typer.echo("Dry run: no changes written.")
    else:
        typer.echo("Saved.")


def main() -> None:  # pragma: no cover - CLI wrapper
    app()


if __name__ == "__main__":
    main()
