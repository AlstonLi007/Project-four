from __future__ import annotations

"""Backfill LightRAG with training examples and artifacts from SQLite."""

import argparse

from db import DEFAULT_DB_PATH
from lightrag_integration import (
    backfill_training_examples_to_rag,
    ensure_index_from_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill LightRAG from wechat_assistant.db (training_examples + artifacts)."
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="rag_store",
        help="LightRAG working_dir (default: rag_store)",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Backfill training_examples into LightRAG.",
    )
    parser.add_argument(
        "--include-artifacts",
        action="store_true",
        help="Backfill artifacts into LightRAG.",
    )
    parser.add_argument(
        "--max-examples-per-contact",
        type=int,
        default=500,
        help="Max training_examples per contact when backfilling.",
    )
    parser.add_argument(
        "--style-profile-version",
        type=int,
        default=0,
        help="Style profile version tag to embed in doc identifiers for A/B analysis.",
    )
    parser.add_argument(
        "--artifacts-kind",
        type=str,
        default=None,
        help="Optional artifacts.kind filter.",
    )
    parser.add_argument(
        "--artifacts-source",
        type=str,
        default=None,
        help="Optional artifacts.source filter.",
    )

    args = parser.parse_args()

    if args.include_training:
        backfill_training_examples_to_rag(
            db_path=args.db,
            working_dir=args.working_dir,
            max_examples_per_contact=args.max_examples_per_contact,
            style_profile_version=args.style_profile_version,
        )

    if args.include_artifacts:
        ensure_index_from_artifacts(
            db_path=args.db,
            working_dir=args.working_dir,
            kind_filter=args.artifacts_kind,
            source_filter=args.artifacts_source,
            style_profile_version=args.style_profile_version,
        )


if __name__ == "__main__":
    main()
