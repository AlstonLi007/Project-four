from __future__ import annotations

"""Small CLI to inspect training_examples for a given contact.

用法示例：

    # 查看某个联系人最近 30 条样本
    python inspect_samples.py --wechat-id 妈妈 --limit 30

    # 查看高权重（比如神回复放大后）的样本
    python inspect_samples.py --wechat-id 妈妈 --min-weight 2.0

    # 打印完整上下文和回复（可能很长）
    python inspect_samples.py --wechat-id 妈妈 --full

默认数据库路径：wechat_assistant.db
"""

import argparse
from datetime import datetime, timezone
from typing import Any, Dict, List

from db import Database


def _age_days(created_at: str) -> float:
    """Compute age in days from an ISO timestamp string."""

    try:
        dt = datetime.fromisoformat(created_at)
    except Exception:
        return 0.0

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    delta = now - dt
    return delta.total_seconds() / 86400.0


def _truncate(text: str, max_len: int = 80) -> str:
    """Truncate long text for compact display."""

    text = text.replace("\n", "\\n")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def inspect_contact_samples(
    db_path: str,
    wechat_id: str,
    *,
    limit: int = 50,
    min_weight: float = 0.0,
    full: bool = False,
) -> None:
    """Print a human-readable view of recent training examples for a contact."""

    db = Database(path=db_path)
    contact = db.get_or_create_contact(wechat_id=wechat_id, display_name=wechat_id)

    examples: List[Dict[str, Any]] = db.get_training_examples(contact_id=contact.id, limit=limit)

    if not examples:
        print(f"[info] contact {wechat_id!r} (id={contact.id}) has no training_examples.")
        return

    filtered = [ex for ex in examples if float(ex.get("weight", 1.0) or 1.0) >= min_weight]
    total = len(examples)

    print(
        f"[summary] contact={wechat_id!r} (id={contact.id}), "
        f"db={db_path}, total_loaded={total}, after_min_weight_filter={len(filtered)}, "
        f"limit={limit}, min_weight={min_weight}"
    )
    print("=" * 80)

    for idx, ex in enumerate(filtered, start=1):
        ctx = (ex.get("context_text") or "").strip()
        rep = (ex.get("reply_text") or "").strip()
        source = ex.get("source") or ""
        weight = float(ex.get("weight", 1.0) or 1.0)
        created_at = ex.get("created_at") or ""
        ex_id = ex.get("id")
        conv_id = ex.get("conversation_id")

        age = _age_days(created_at) if created_at else 0.0

        print(f"[{idx}] example_id={ex_id} conv_id={conv_id}")
        print(f"  created_at : {created_at}  (age={age:.1f} days)")
        print(f"  weight     : {weight:.4f}")
        print(f"  source     : {source}")

        if full:
            print("  context    :")
            for line in ctx.splitlines():
                print(f"    {line}")
            print("  reply      :")
            for line in rep.splitlines():
                print(f"    {line}")
        else:
            print(f"  context    : {_truncate(ctx)}")
            print(f"  reply      : {_truncate(rep)}")

        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect training_examples for a single contact."
    )
    parser.add_argument(
        "--db",
        type=str,
        default="wechat_assistant.db",
        help="Path to SQLite DB (default: wechat_assistant.db)",
    )
    parser.add_argument(
        "--wechat-id",
        type=str,
        required=True,
        help="WeChat identifier for the contact (same as used in training).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of examples to load (default: 50).",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.0,
        help="Filter out examples whose weight is below this threshold.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Print full context / reply text instead of truncated preview.",
    )

    args = parser.parse_args()

    inspect_contact_samples(
        db_path=args.db,
        wechat_id=args.wechat_id,
        limit=args.limit,
        min_weight=args.min_weight,
        full=args.full,
    )


if __name__ == "__main__":
    main()
