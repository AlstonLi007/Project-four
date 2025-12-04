from __future__ import annotations

"""Helpers to correlate Memori-tracked LLM calls with local training data.

This module pulls data from two sources that share the same SQLite file by
default:

* Memori event tables (for LLM calls instrumented via ``memori_integration``)
* Local assistant tables defined in ``db.py`` (training examples, contacts)

Use these helpers from notebooks or scripts to build per-contact summaries or
visualize which prompts produce long replies. The Memori table/column names are
configurable via ``MemoriEventTableConfig`` so you can align to the actual
schema present in your deployment.
"""

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text

from db import DEFAULT_DB_PATH

DEFAULT_DB_URL = os.environ.get("MEMORI_DB_URL", f"sqlite:///{DEFAULT_DB_PATH}")


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------

def get_engine(db_url: Optional[str] = None):
    """Return a SQLAlchemy engine pointed at the shared SQLite database."""

    url = db_url or DEFAULT_DB_URL
    return create_engine(url, future=True)


# ---------------------------------------------------------------------------
# Local training data (assistant tables)
# ---------------------------------------------------------------------------

def load_training_examples(engine) -> pd.DataFrame:
    """Return training examples joined with contact metadata.

    Columns include ``context_text``, ``reply_text``, ``source``, ``weight``,
    ``created_at``, plus ``wechat_id`` and ``display_name`` pulled from
    ``contacts``.
    """

    query = text(
        """
        SELECT
            te.id,
            te.contact_id,
            te.conversation_id,
            te.context_text,
            te.reply_text,
            te.source,
            te.weight,
            te.created_at,
            c.wechat_id,
            c.display_name
        FROM training_examples AS te
        JOIN contacts AS c ON te.contact_id = c.id
        """
    )
    with engine.connect() as conn:
        return pd.read_sql_query(query, conn)


# ---------------------------------------------------------------------------
# Memori events (LLM calls)
# ---------------------------------------------------------------------------


@dataclass
class MemoriEventTableConfig:
    """Table/column mapping for Memori events.

    Adjust the defaults to match the Memori schema in your environment.
    """

    table: str = "memori_events"
    col_entity_id: str = "entity_id"
    col_process_id: str = "process_id"
    col_created_at: str = "created_at"
    col_model: str = "model"
    col_input: str = "prompt_text"
    col_output: str = "completion_text"
    col_tags: str = "tags_json"
    col_latency_ms: str = "latency_ms"


def _safe_json(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except Exception:
        return {}


def load_memori_events(
    engine,
    *,
    cfg: MemoriEventTableConfig = MemoriEventTableConfig(),
    process_id: str = "wechat_auto_reply",
) -> pd.DataFrame:
    """Load Memori-tracked LLM events for a process ID.

    The returned DataFrame includes parsed tag fields plus basic length/latency
    metrics for quick plotting.
    """

    columns = {
        "entity_id": cfg.col_entity_id,
        "process_id": cfg.col_process_id,
        "created_at": cfg.col_created_at,
        "model": cfg.col_model,
        "prompt_raw": cfg.col_input,
        "completion_raw": cfg.col_output,
        "tags_raw": cfg.col_tags,
        "latency_ms": cfg.col_latency_ms,
    }

    select_parts = [
        f"{sql_col} AS {alias}" for alias, sql_col in columns.items()
    ]
    query = f"""
        SELECT {', '.join(select_parts)}
        FROM {cfg.table}
        WHERE {cfg.col_process_id} = :pid
    """

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn, params={"pid": process_id})

    if df.empty:
        return df

    df["prompt_chars"] = df["prompt_raw"].astype(str).str.len()
    df["completion_chars"] = df["completion_raw"].astype(str).str.len()

    tags = df["tags_raw"].apply(_safe_json)
    df["tag_channel"] = tags.apply(lambda d: d.get("channel"))
    df["tag_wechat_id"] = tags.apply(lambda d: d.get("wechat_id"))
    df["tag_display_name"] = tags.apply(lambda d: d.get("display_name"))
    df["tag_style_label"] = tags.apply(lambda d: d.get("style_label"))
    df["tag_prompt_version"] = tags.apply(lambda d: d.get("prompt_version"))
    df["tag_style_profile_model_id"] = tags.apply(lambda d: d.get("style_profile_model_id"))
    df["tag_conversation_id"] = tags.apply(lambda d: d.get("conversation_id"))
    df["tag_message_id"] = tags.apply(lambda d: d.get("message_id"))
    df["tag_style_profile_id"] = tags.apply(lambda d: d.get("style_profile_id"))
    df["tag_style_profile_version"] = tags.apply(lambda d: d.get("style_profile_version"))
    df["tag_rag_enabled"] = tags.apply(lambda d: d.get("rag_enabled"))
    df["tag_rag_profile_id"] = tags.apply(lambda d: d.get("rag_profile_id"))
    df["tag_rag_track_id"] = tags.apply(lambda d: d.get("rag_track_id"))
    df["tag_rag_engine"] = tags.apply(lambda d: d.get("rag_engine"))
    df["tag_rag_mode"] = tags.apply(lambda d: d.get("rag_mode"))
    df["tag_rag_hit_count"] = tags.apply(lambda d: d.get("rag_hit_count"))
    df["tag_rag_doc_ids"] = tags.apply(lambda d: d.get("rag_doc_ids"))

    return df


# ---------------------------------------------------------------------------
# Conversation/message-aligned pairs
# ---------------------------------------------------------------------------


def load_llm_call_pairs(engine) -> pd.DataFrame:
    """Return Memori calls joined to messages and candidates by ID tags.

    Each row represents one LLM call (Memori event) aligned with the triggering
    message and any reply candidates recorded for that message. This requires
    that ``brain._call_llm`` tagged ``conversation_id`` and ``message_id`` in
    Memori attribution metadata, which the current version does.
    """

    sql = text(
        """
        SELECT
            me.id                 AS memori_event_id,
            me.created_at         AS llm_called_at,
            me.model              AS llm_model,
            me.latency_ms         AS llm_latency_ms,

            json_extract(me.tags_json, '$.wechat_id')        AS wechat_id,
            json_extract(me.tags_json, '$.conversation_id')  AS conversation_id,
            json_extract(me.tags_json, '$.message_id')       AS message_id,
            json_extract(me.tags_json, '$.style_label')      AS style_label,
            json_extract(me.tags_json, '$.style_profile_id') AS style_profile_id,
            json_extract(me.tags_json, '$.style_profile_version') AS style_profile_version,
            json_extract(me.tags_json, '$.style_profile_model_id') AS style_profile_model_id,
            json_extract(me.tags_json, '$.prompt_version')   AS prompt_version,
            json_extract(me.tags_json, '$.rag_enabled')      AS rag_enabled,
            json_extract(me.tags_json, '$.rag_profile_id')   AS rag_profile_id,
            json_extract(me.tags_json, '$.rag_track_id')     AS rag_track_id,
            json_extract(me.tags_json, '$.rag_engine')       AS rag_engine,
            json_extract(me.tags_json, '$.rag_mode')         AS rag_mode,
            json_extract(me.tags_json, '$.rag_hit_count')    AS rag_hit_count,
            json_extract(me.tags_json, '$.rag_doc_ids')      AS rag_doc_ids,

            c.display_name        AS contact_display_name,
            msg.sender            AS msg_sender,
            msg.raw_text          AS msg_text,
            msg.created_at        AS msg_created_at,

            rc.id                 AS candidate_id,
            rc.candidate_index    AS candidate_index,
            rc.text               AS candidate_text,
            rc.chosen             AS candidate_chosen,
            rc.chosen_at          AS candidate_chosen_at,
            rc.edited_text        AS candidate_edited_text

        FROM memori_events AS me
        JOIN messages AS msg
          ON msg.id = CAST(json_extract(me.tags_json, '$.message_id') AS INTEGER)
        JOIN conversations AS conv
          ON conv.id = msg.conversation_id
        JOIN contacts AS c
          ON c.id = conv.contact_id
        LEFT JOIN reply_candidates AS rc
          ON rc.message_id = msg.id
        WHERE me.process_id = 'wechat_auto_reply'
        """
    )

    with engine.connect() as conn:
        return pd.read_sql_query(sql, conn)


def ensure_llm_pairs_view(engine) -> None:
    """Create a SQLite view aligning Memori events with messages/candidates.

    The view is named ``v_wechat_llm_pairs`` and mirrors the query used by
    :func:`load_llm_call_pairs`. It is safe to call multiple times.
    """

    create_sql = text(
        """
        CREATE VIEW IF NOT EXISTS v_wechat_llm_pairs AS
        SELECT
            me.id                 AS memori_event_id,
            me.created_at         AS llm_called_at,
            me.model              AS llm_model,
            me.latency_ms         AS llm_latency_ms,

            json_extract(me.tags_json, '$.wechat_id')        AS wechat_id,
            json_extract(me.tags_json, '$.conversation_id')  AS conversation_id,
            json_extract(me.tags_json, '$.message_id')       AS message_id,
            json_extract(me.tags_json, '$.style_label')      AS style_label,
            json_extract(me.tags_json, '$.style_profile_id') AS style_profile_id,
            json_extract(me.tags_json, '$.style_profile_version') AS style_profile_version,
            json_extract(me.tags_json, '$.style_profile_model_id') AS style_profile_model_id,
            json_extract(me.tags_json, '$.prompt_version')   AS prompt_version,
            json_extract(me.tags_json, '$.rag_enabled')      AS rag_enabled,
            json_extract(me.tags_json, '$.rag_profile_id')   AS rag_profile_id,
            json_extract(me.tags_json, '$.rag_track_id')     AS rag_track_id,
            json_extract(me.tags_json, '$.rag_engine')       AS rag_engine,
            json_extract(me.tags_json, '$.rag_mode')         AS rag_mode,
            json_extract(me.tags_json, '$.rag_hit_count')    AS rag_hit_count,
            json_extract(me.tags_json, '$.rag_doc_ids')      AS rag_doc_ids,

            c.display_name        AS contact_display_name,
            msg.sender            AS msg_sender,
            msg.raw_text          AS msg_text,
            msg.created_at        AS msg_created_at,

            rc.id                 AS candidate_id,
            rc.candidate_index    AS candidate_index,
            rc.text               AS candidate_text,
            rc.chosen             AS candidate_chosen,
            rc.chosen_at          AS candidate_chosen_at,
            rc.edited_text        AS candidate_edited_text

        FROM memori_events AS me
        JOIN messages AS msg
          ON msg.id = CAST(json_extract(me.tags_json, '$.message_id') AS INTEGER)
        JOIN conversations AS conv
          ON conv.id = msg.conversation_id
        JOIN contacts AS c
          ON c.id = conv.contact_id
        LEFT JOIN reply_candidates AS rc
          ON rc.message_id = msg.id
        WHERE me.process_id = 'wechat_auto_reply'
        """
    )

    with engine.begin() as conn:
        conn.execute(create_sql)


# ---------------------------------------------------------------------------
# Dashboard views
# ---------------------------------------------------------------------------


def ensure_llm_dashboard_views(engine, *, long_threshold: int = 80) -> None:
    """Create enriched candidate and daily metric views for dashboarding.

    Views created:
        - ``v_wechat_llm_candidate_metrics``: candidate-level facts with helper
          booleans/lengths for quick aggregation.
        - ``v_wechat_llm_metrics_daily``: date-grained rollup across style
          versions, prompt versions, and RAG switches with length/selection
          metrics.
        - ``v_wechat_llm_metrics_style_overall``: style/prompt-level summary
          weighted by candidate counts across days.
        - ``v_wechat_llm_metrics_contact_overall``: per-contact/prompt summary
          weighted by candidate counts across days.
        - ``v_wechat_llm_failover_daily``: heuristic failover counts per day and
          contact/backend for visibility into outages.
    """

    # Ensure the base alignment view exists first.
    ensure_llm_pairs_view(engine)

    candidate_view_sql = f"""
    CREATE VIEW IF NOT EXISTS v_wechat_llm_candidate_metrics AS
    SELECT
        p.memori_event_id,
        p.llm_called_at,
        date(p.llm_called_at)                  AS llm_date,
        p.llm_model,
        p.llm_latency_ms,

        p.wechat_id,
        p.contact_display_name,
        p.conversation_id,
        p.message_id,

        p.style_profile_id,
        p.style_label,
        p.style_profile_version,
        p.style_profile_model_id,
        p.prompt_version,

        p.rag_enabled,
        p.rag_profile_id,
        p.rag_track_id,
        p.rag_engine,
        p.rag_mode,
        p.rag_hit_count,
        p.rag_doc_ids,

        p.candidate_id,
        p.candidate_index,
        p.candidate_text,
        p.candidate_chosen,
        p.candidate_chosen_at,
        p.candidate_edited_text,

        length(coalesce(p.candidate_text, '')) AS candidate_len,
        CASE
            WHEN length(coalesce(p.candidate_text, '')) > {long_threshold}
            THEN 1 ELSE 0
        END AS is_long,
        CASE
            WHEN p.candidate_chosen = 1 THEN 1 ELSE 0
        END AS is_chosen,
        CASE
            WHEN p.candidate_chosen = 1
                 AND trim(ifnull(p.candidate_edited_text, '')) <> ''
            THEN 1
            ELSE 0
        END AS is_edited
    FROM v_wechat_llm_pairs AS p
    WHERE p.candidate_id IS NOT NULL
    """

    daily_view_sql = """
    CREATE VIEW IF NOT EXISTS v_wechat_llm_metrics_daily AS
    SELECT
        llm_date,

        wechat_id,
        contact_display_name,

        style_profile_id,
        style_label,
        style_profile_version,
        style_profile_model_id,
        prompt_version,

        rag_enabled,
        rag_profile_id,
        rag_track_id,
        rag_engine,
        rag_mode,

        COUNT(*)                           AS total_candidates,
        COUNT(DISTINCT memori_event_id)    AS llm_call_count,
        COUNT(DISTINCT message_id)         AS message_count,

        AVG(candidate_len)                 AS avg_len,
        AVG(is_long)                       AS long_ratio,
        AVG(is_chosen)                     AS chosen_ratio,
        AVG(is_edited)                     AS edited_ratio_overall,

        CASE
            WHEN SUM(is_chosen) > 0 THEN 1.0 * SUM(is_edited) / SUM(is_chosen)
            ELSE NULL
        END                                AS edited_ratio_given_chosen,

        AVG(llm_latency_ms)                AS avg_latency_ms,
        AVG(COALESCE(rag_hit_count, 0))    AS avg_rag_hit_count

    FROM v_wechat_llm_candidate_metrics
    GROUP BY
        llm_date,
        wechat_id,
        contact_display_name,
        style_profile_id,
        style_label,
        style_profile_version,
        style_profile_model_id,
        prompt_version,
        rag_enabled,
        rag_engine,
        rag_mode,
        rag_profile_id,
        rag_track_id
    """

    style_overall_sql = """
    CREATE VIEW IF NOT EXISTS v_wechat_llm_metrics_style_overall AS
    SELECT
        style_profile_id,
        style_label,
        style_profile_version,
        style_profile_model_id,
        prompt_version,

        SUM(total_candidates)                         AS total_candidates,
        SUM(message_count)                            AS total_messages,
        SUM(llm_call_count)                           AS total_llm_calls,

        SUM(avg_len * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS avg_len,
        SUM(long_ratio * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS long_ratio,
        SUM(chosen_ratio * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS chosen_ratio,
        SUM(edited_ratio_overall * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS edited_ratio_overall,
        SUM(edited_ratio_given_chosen * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS edited_ratio_given_chosen,

        AVG(avg_latency_ms)                           AS avg_latency_ms,
        AVG(avg_rag_hit_count)                        AS avg_rag_hit_count
    FROM v_wechat_llm_metrics_daily
    GROUP BY
        style_profile_id,
        style_label,
        style_profile_version,
        style_profile_model_id,
        prompt_version
    """

    contact_overall_sql = """
    CREATE VIEW IF NOT EXISTS v_wechat_llm_metrics_contact_overall AS
    SELECT
        wechat_id,
        contact_display_name,
        prompt_version,

        SUM(total_candidates)                         AS total_candidates,
        SUM(message_count)                            AS total_messages,
        SUM(llm_call_count)                           AS total_llm_calls,

        SUM(avg_len * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS avg_len,
        SUM(long_ratio * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS long_ratio,
        SUM(chosen_ratio * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS chosen_ratio,
        SUM(edited_ratio_overall * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS edited_ratio_overall,
        SUM(edited_ratio_given_chosen * total_candidates)
            / NULLIF(SUM(total_candidates), 0)        AS edited_ratio_given_chosen,

        AVG(avg_latency_ms)                           AS avg_latency_ms,
        AVG(avg_rag_hit_count)                        AS avg_rag_hit_count
    FROM v_wechat_llm_metrics_daily
    GROUP BY
        wechat_id,
        contact_display_name,
        prompt_version
    """

    failover_sql = """
    CREATE VIEW IF NOT EXISTS v_wechat_llm_failover_daily AS
    SELECT
        date(ts) AS llm_date,
        c.wechat_id,
        c.display_name AS contact_display_name,
        f.contact_id,
        f.backend,
        f.heuristic_name,
        COUNT(*) AS fail_count
    FROM llm_failover_events AS f
    LEFT JOIN contacts AS c ON c.id = f.contact_id
    GROUP BY llm_date, c.wechat_id, c.display_name, f.contact_id, f.backend, f.heuristic_name
    """

    with engine.begin() as conn:
        conn.execute(text(candidate_view_sql))
        conn.execute(text(daily_view_sql))
        conn.execute(text(style_overall_sql))
        conn.execute(text(contact_overall_sql))
        conn.execute(text(failover_sql))

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def style_profile_metrics(
    engine,
    *,
    long_threshold: int = 80,
) -> pd.DataFrame:
    """Compute per-style metrics (length, long ratio, chosen/edited rates).

    Returns a DataFrame with columns such as ``total_candidates``,
    ``avg_len``, ``long_ratio``, ``chosen_ratio``, ``edited_ratio_overall`` and
    ``edited_ratio_given_chosen`` grouped by ``style_profile_id``,
    ``style_label`` and ``prompt_version``.
    """

    pairs = load_llm_call_pairs(engine)
    if pairs.empty:
        return pairs

    df = pairs[pairs["candidate_text"].notna()].copy()
    df["len"] = df["candidate_text"].astype(str).str.len()
    df["is_long"] = df["len"] > long_threshold
    df["is_chosen"] = df["candidate_chosen"] == 1
    df["is_edited"] = df["is_chosen"] & df["candidate_edited_text"].fillna("").str.strip().ne("")

    group_cols = ["style_profile_id", "style_label", "style_profile_version", "prompt_version"]
    agg = (
        df.groupby(group_cols)
        .agg(
            total_candidates=("candidate_id", "count"),
            avg_len=("len", "mean"),
            long_ratio=("is_long", "mean"),
            chosen_ratio=("is_chosen", "mean"),
            edited_ratio_overall=("is_edited", "mean"),
        )
        .reset_index()
    )

    # Compute edited rate among chosen only
    chosen_counts = df.groupby(group_cols)["is_chosen"].sum().rename("chosen_count")
    edited_chosen = df[df["is_edited"]].groupby(group_cols)["is_edited"].sum().rename(
        "edited_chosen_count"
    )
    agg = agg.merge(chosen_counts, on=group_cols, how="left").merge(
        edited_chosen, on=group_cols, how="left"
    )
    agg[["chosen_count", "edited_chosen_count"]] = agg[["chosen_count", "edited_chosen_count"]].fillna(0)
    agg["edited_ratio_given_chosen"] = agg.apply(
        lambda row: (row.get("edited_chosen_count", 0) or 0)
        / max(row.get("chosen_count", 0), 1),
        axis=1,
    )

    return agg.sort_values(group_cols)


def contact_prompt_version_metrics(
    engine,
    *,
    long_threshold: int = 80,
) -> pd.DataFrame:
    """Compute per-contact, per-prompt-version metrics for candidate quality."""

    pairs = load_llm_call_pairs(engine)
    if pairs.empty:
        return pairs

    df = pairs[pairs["candidate_text"].notna()].copy()
    df["len"] = df["candidate_text"].astype(str).str.len()
    df["is_long"] = df["len"] > long_threshold
    df["is_chosen"] = df["candidate_chosen"] == 1
    df["is_edited"] = df["is_chosen"] & df["candidate_edited_text"].fillna("").str.strip().ne("")

    group_cols = ["wechat_id", "contact_display_name", "prompt_version"]
    agg = (
        df.groupby(group_cols)
        .agg(
            llm_call_count=("memori_event_id", "nunique"),
            message_count=("message_id", "nunique"),
            total_candidates=("candidate_id", "count"),
            avg_len=("len", "mean"),
            long_ratio=("is_long", "mean"),
            chosen_ratio=("is_chosen", "mean"),
            edited_ratio=("is_edited", "mean"),
            avg_latency_ms=("llm_latency_ms", "mean"),
        )
        .reset_index()
    )

    return agg.sort_values(group_cols)


def style_profile_before_after(
    engine,
    *,
    long_threshold: int = 80,
) -> pd.DataFrame:
    """Compare metrics before/after the most recent style_profile update."""

    pairs = load_llm_call_pairs(engine)
    if pairs.empty:
        return pairs

    df = pairs[pairs["candidate_text"].notna()].copy()
    df["len"] = df["candidate_text"].astype(str).str.len()
    df["is_long"] = df["len"] > long_threshold
    df["is_chosen"] = df["candidate_chosen"] == 1
    df["is_edited"] = df["is_chosen"] & df["candidate_edited_text"].fillna("").str.strip().ne("")
    df["llm_called_at"] = pd.to_datetime(df["llm_called_at"])

    profiles = pd.read_sql(
        text(
            """
            SELECT id, label, COALESCE(updated_at, created_at) AS updated_at
            FROM style_profiles
            """
        ),
        engine,
    )

    if profiles.empty:
        return pd.DataFrame()

    profiles["updated_at"] = pd.to_datetime(profiles["updated_at"])
    df["style_profile_id_int"] = pd.to_numeric(
        df["style_profile_id"], errors="coerce"
    ).astype("Int64")

    merged = df.merge(
        profiles,
        left_on="style_profile_id_int",
        right_on="id",
        how="inner",
        suffixes=("", "_profile"),
    )

    merged["period"] = merged.apply(
        lambda row: "before"
        if row["llm_called_at"] < row["updated_at"]
        else "after",
        axis=1,
    )

    group_cols = ["id", "label", "period"]
    agg = (
        merged.groupby(group_cols)
        .agg(
            total_candidates=("candidate_id", "count"),
            n_messages=("message_id", "nunique"),
            avg_len=("len", "mean"),
            long_ratio=("is_long", "mean"),
            chosen_ratio=("is_chosen", "mean"),
            edited_ratio=("is_edited", "mean"),
        )
        .reset_index()
        .rename(columns={"id": "style_profile_id"})
    )

    return agg.sort_values(["style_profile_id", "period"])


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------

def build_contact_summary(memori_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-contact summary combining Memori events and training data."""

    if not memori_df.empty:
        agg_memori = (
            memori_df.groupby("tag_wechat_id")
            .agg(
                memori_calls=("tag_wechat_id", "size"),
                memori_avg_completion_chars=("completion_chars", "mean"),
                memori_p90_completion_chars=("completion_chars", lambda s: s.quantile(0.9)),
                memori_long_reply_ratio=("completion_chars", lambda s: (s > 80).mean()),
                memori_avg_latency_ms=("latency_ms", "mean"),
                last_llm_call_at=("created_at", "max"),
            )
            .reset_index()
            .rename(columns={"tag_wechat_id": "wechat_id"})
        )
    else:
        agg_memori = pd.DataFrame(
            columns=[
                "wechat_id",
                "memori_calls",
                "memori_avg_completion_chars",
                "memori_p90_completion_chars",
                "memori_long_reply_ratio",
                "memori_avg_latency_ms",
                "last_llm_call_at",
            ]
        )

    if not train_df.empty:
        agg_train = (
            train_df.groupby("wechat_id")
            .agg(
                training_examples=("id", "size"),
                training_avg_reply_len=("reply_text", lambda s: s.astype(str).str.len().mean()),
                training_p90_reply_len=("reply_text", lambda s: s.astype(str).str.len().quantile(0.9)),
                training_avg_weight=("weight", "mean"),
                last_training_at=("created_at", "max"),
                display_name=("display_name", "first"),
            )
            .reset_index()
        )
    else:
        agg_train = pd.DataFrame(
            columns=[
                "wechat_id",
                "training_examples",
                "training_avg_reply_len",
                "training_p90_reply_len",
                "training_avg_weight",
                "last_training_at",
                "display_name",
            ]
        )

    summary = pd.merge(
        agg_train,
        agg_memori,
        how="outer",
        on="wechat_id",
        suffixes=("_train", "_memori"),
    ).fillna({"training_examples": 0, "memori_calls": 0})

    summary["llm_chars_per_example"] = summary["memori_avg_completion_chars"] / summary["training_examples"].clip(lower=1)

    return summary


__all__ = [
    "DEFAULT_DB_URL",
    "MemoriEventTableConfig",
    "build_contact_summary",
    "ensure_llm_dashboard_views",
    "ensure_llm_pairs_view",
    "get_engine",
    "contact_prompt_version_metrics",
    "load_llm_call_pairs",
    "load_memori_events",
    "load_training_examples",
    "style_profile_before_after",
    "style_profile_metrics",
]
