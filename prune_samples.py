from __future__ import annotations

"""CLI to prune old & low-weight training_examples so the DB behaves like a rolling window.

设计原则：
    - 只删除「又老又轻」的样本：age_days > older_than_days AND weight < max_weight
    - 每个 contact 至少保留最近 min_keep 条，不管老不老，防止一刀切删光
    - 支持 dry-run，先看将要删除什么再确认

用法示例：

    # 清理某个联系人：只删 180 天前且权重 < 1.0 且不在最近 100 条里的样本（真实删除）
    python prune_samples.py --wechat-id 妈妈 --older-than-days 180 --max-weight 1.0 --min-keep 100

    # 针对全库所有联系人跑一遍，但只预览不删
    python prune_samples.py --older-than-days 365 --max-weight 0.8 --dry-run

参数含义：
    --db              SQLite 路径，默认 wechat_assistant.db
    --wechat-id       只处理某一个 contact；不传则遍历所有 contact
    --older-than-days 样本年龄阈值（天），大于此值才算「老」
    --max-weight      权重阈值，小于此权重才算「弱」
    --min-keep        每个 contact 至少保留的最新样本数（按 created_at 排序）
    --dry-run         只打印将要删除的内容，不真正删除
"""

import argparse
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

import sqlite3

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


def _iter_contacts(db: Database) -> Iterable[Dict[str, Any]]:
    """Yield all contacts (id, wechat_id, display_name)."""

    with sqlite3.connect(str(db.path)) as con:
        con.row_factory = sqlite3.Row
        cur = con.execute("SELECT id, wechat_id, display_name FROM contacts ORDER BY id ASC")
        for row in cur.fetchall():
            yield dict(row)


def _load_examples_for_contact(
    con: sqlite3.Connection,
    contact_id: int,
) -> List[Dict[str, Any]]:
    """Load all training_examples for a contact, newest first."""

    con.row_factory = sqlite3.Row
    cur = con.execute(
        """
        SELECT id, contact_id, conversation_id,
               context_text, reply_text,
               source, weight, created_at
        FROM training_examples
        WHERE contact_id = ?
        ORDER BY datetime(created_at) DESC, id DESC
        """,
        (contact_id,),
    )
    rows = cur.fetchall()
    return [dict(r) for r in rows]


def _plan_deletions_for_contact(
    examples: List[Dict[str, Any]],
    *,
    older_than_days: float,
    max_weight: float,
    min_keep: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Decide which example IDs to delete for a single contact."""

    to_delete: List[int] = []

    total = len(examples)
    if total == 0:
        return [], {"total": 0, "kept_recent": 0, "candidates": 0}

    kept_recent = min(total, max(0, min_keep))

    for idx, ex in enumerate(examples):
        ex_id = ex["id"]
        weight = float(ex.get("weight", 1.0) or 1.0)
        created_at = ex.get("created_at") or ""
        age = _age_days(created_at) if created_at else 0.0

        # 保底：前 min_keep 条（最新）强制保留
        if idx < min_keep:
            continue

        too_old = age > older_than_days
        too_light = weight < max_weight

        if too_old and too_light:
            to_delete.append(int(ex_id))

    stats = {
        "total": total,
        "kept_recent": kept_recent,
        "candidates": len(to_delete),
    }
    return to_delete, stats


def prune_for_contact(
    db_path: str,
    contact_row: Dict[str, Any],
    *,
    older_than_days: float,
    max_weight: float,
    min_keep: int,
    dry_run: bool,
) -> None:
    """Plan & optionally execute pruning for a single contact."""

    contact_id = int(contact_row["id"])
    wechat_id = contact_row["wechat_id"]
    display_name = contact_row.get("display_name") or wechat_id

    with sqlite3.connect(db_path) as con:
        examples = _load_examples_for_contact(con, contact_id=contact_id)

        to_delete, stats = _plan_deletions_for_contact(
            examples,
            older_than_days=older_than_days,
            max_weight=max_weight,
            min_keep=min_keep,
        )

        total = stats["total"]
        candidates = stats["candidates"]

        if total == 0:
            print(f"[{wechat_id}] ({display_name}) has no training_examples.")
            return

        print(
            f"[{wechat_id}] ({display_name}) total={total}, "
            f"kept_recent={stats['kept_recent']}, "
            f"delete_candidates={candidates}"
        )

        if not to_delete:
            return

        # 为了 debug，预览前几条要删的样本
        preview = to_delete[:5]
        print(f"  -> first {len(preview)} candidate ids: {preview}")

        if dry_run:
            print("  [dry-run] no deletions executed.")
            return

        # 真正删除
        q_marks = ",".join("?" for _ in to_delete)
        con.execute(f"DELETE FROM training_examples WHERE id IN ({q_marks})", to_delete)
        con.commit()
        print(f"  [done] deleted {len(to_delete)} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prune old & low-weight training_examples for one or all contacts."
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
        default=None,
        help="Limit pruning to a single contact by wechat_id. If omitted, all contacts are processed.",
    )
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=365.0,
        help="Age threshold in days; samples older than this AND below max_weight may be pruned (default: 365).",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=1.0,
        help="Weight threshold; samples with weight < max-weight are considered low-weight (default: 1.0).",
    )
    parser.add_argument(
        "--min-keep",
        type=int,
        default=100,
        help="Keep at least this many most-recent samples per contact, regardless of age/weight (default: 100).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting anything.",
    )

    args = parser.parse_args()

    db = Database(path=args.db)

    if args.wechat_id:
        # 确保 contact 存在 / 拿到 id
        contact = db.get_or_create_contact(wechat_id=args.wechat_id, display_name=args.wechat_id)
        row = {
            "id": contact.id,
            "wechat_id": contact.wechat_id,
            "display_name": contact.display_name,
        }
        prune_for_contact(
            db_path=args.db,
            contact_row=row,
            older_than_days=args.older_than_days,
            max_weight=args.max_weight,
            min_keep=args.min_keep,
            dry_run=args.dry_run,
        )
    else:
        for contact_row in _iter_contacts(db):
            prune_for_contact(
                db_path=args.db,
                contact_row=contact_row,
                older_than_days=args.older_than_days,
                max_weight=args.max_weight,
                min_keep=args.min_keep,
                dry_run=args.dry_run,
            )


if __name__ == "__main__":
    main()
