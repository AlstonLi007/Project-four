from __future__ import annotations

"""LightRAG wrapper for WeChat drafting.

This module centralizes retrieval toggles so ``brain.py`` can cleanly opt into
(or out of) retrieval-augmented prompting while keeping Memori analytics aware
of which configuration was used.
"""

import os
from typing import Any, Dict, Tuple

try:  # LightRAG may be optional in some environments
    from lightrag_integration import rag_query_with_meta_for_contact
except Exception:  # pragma: no cover - optional dependency safeguard
    rag_query_with_meta_for_contact = None  # type: ignore

USE_WECHAT_RAG = os.environ.get("USE_WECHAT_RAG", os.environ.get("USE_RAG", "0")) == "1"
WECHAT_RAG_TOP_K = int(os.environ.get("WECHAT_RAG_TOP_K", "4"))
WECHAT_RAG_MODE = os.environ.get("WECHAT_RAG_MODE", None)
DEFAULT_RAG_HALF_LIFE_DAYS = float(os.environ.get("WECHAT_RAG_HALF_LIFE_DAYS", 30.0))


def get_rag_context(
    wechat_id: str,
    latest_text: str,
    *,
    style_profile_version: str | None = None,
    rag_mode: str | None = None,
    half_life_days: float | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Return a prompt-ready RAG context and metadata for Memori tags.

    The metadata is prefixed with ``rag_`` keys when passed into Memori tags so
    downstream analytics can group by retrieval usage. When RAG is disabled or
    unavailable, an empty context and a metadata block explaining the reason
    are returned.
    """

    latest_clean = (latest_text or "").strip()
    resolved_mode = (rag_mode or WECHAT_RAG_MODE or "off").lower() if rag_mode or WECHAT_RAG_MODE else "off"
    resolved_half_life = half_life_days if half_life_days is not None else DEFAULT_RAG_HALF_LIFE_DAYS

    if not USE_WECHAT_RAG or resolved_mode == "off":
        return "", {
            "enabled": False,
            "engine": "LightRAG",
            "reason": "rag_disabled",
            "rag_mode": resolved_mode,
            "half_life_days": resolved_half_life,
        }
    if not latest_clean:
        return "", {
            "enabled": False,
            "engine": "LightRAG",
            "reason": "empty_query",
            "rag_mode": resolved_mode,
            "half_life_days": resolved_half_life,
        }
    if rag_query_with_meta_for_contact is None:
        return "", {
            "enabled": False,
            "engine": "LightRAG",
            "reason": "lightrag_missing",
            "rag_mode": resolved_mode,
            "half_life_days": resolved_half_life,
        }

    try:
        top_k = WECHAT_RAG_TOP_K
        if resolved_mode == "light":
            top_k = max(1, min(WECHAT_RAG_TOP_K, 3))
        elif resolved_mode == "aggressive":
            top_k = max(WECHAT_RAG_TOP_K, 8)

        context, meta = rag_query_with_meta_for_contact(
            wechat_id,
            latest_clean,
            top_k=top_k,
            style_profile_version=style_profile_version,
        )
        base_meta = {
            "enabled": True,
            "engine": "LightRAG",
            "mode": resolved_mode or WECHAT_RAG_MODE,
            "top_k": top_k,
            "style_profile_version": style_profile_version,
            "rag_mode": resolved_mode,
            "half_life_days": resolved_half_life,
        }
        if meta:
            base_meta.update(meta)
        return context, base_meta
    except Exception as exc:  # pragma: no cover - defensive fallback
        return "", {
            "enabled": False,
            "engine": "LightRAG",
            "error": repr(exc),
            "top_k": WECHAT_RAG_TOP_K,
            "style_profile_version": style_profile_version,
            "rag_mode": resolved_mode,
            "half_life_days": resolved_half_life,
        }
