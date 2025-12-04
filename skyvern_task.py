from __future__ import annotations

"""CLI helper to launch a Skyvern run and log the result to artifacts."""

import argparse
import json
from typing import Any, Dict

from db import Database, DEFAULT_DB_PATH
from skyvern_integration import SkyvernClient, log_skyvern_artifact


def run_skyvern_task(
    *,
    db_path: str,
    start_url: str,
    instructions: str,
    kind: str = "skyvern_run",
    source: str = "skyvern",
    extra_meta: Dict[str, Any] | None = None,
) -> int:
    """Create a Skyvern run and store the raw payload in ``artifacts``.

    Returns:
        The inserted artifact ID for traceability.
    """

    client = SkyvernClient()
    db = Database(path=db_path)

    run = client.create_run(start_url=start_url, instructions=instructions, metadata=extra_meta or {})
    print(f"[skyvern] created run={run.run_id} status={run.status}")

    title = f"Skyvern run {run.run_id}"
    text_content = json.dumps(run.raw, ensure_ascii=False, indent=2)
    meta = {
        "skyvern_run_id": run.run_id,
        "status": run.status,
        "output_url": run.output_url,
    }
    if extra_meta:
        meta["extra"] = extra_meta

    artifact_id = log_skyvern_artifact(
        db,
        kind=kind,
        source=source,
        title=title,
        text_content=text_content,
        meta=meta,
        url=run.output_url,
    )
    print(f"[skyvern] stored artifact_id={artifact_id}")
    return artifact_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kick off a Skyvern browser task and store the result as an artifact.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument("--start-url", type=str, required=True, help="Start URL for the Skyvern run.")
    parser.add_argument(
        "--instructions",
        type=str,
        required=True,
        help="High-level natural language instructions for Skyvern.",
    )
    parser.add_argument("--kind", type=str, default="skyvern_run", help="artifacts.kind value")
    parser.add_argument("--source", type=str, default="skyvern", help="artifacts.source value")
    args = parser.parse_args()

    run_skyvern_task(
        db_path=args.db,
        start_url=args.start_url,
        instructions=args.instructions,
        kind=args.kind,
        source=args.source,
    )


if __name__ == "__main__":
    main()
