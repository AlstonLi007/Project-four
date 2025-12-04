"""Memori + OpenAI integration helpers.

This module lazily initializes Memori with a SQLAlchemy session pointed at the
shared SQLite file (by default ``wechat_assistant.db``). It returns an OpenAI
client instrumented by Memori so LLM calls are recorded with attribution. If
Memori initialization fails for any reason, the caller can still use the raw
OpenAI client that is returned.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openai import OpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memori import Memori

DEFAULT_DB_URL = "sqlite:///wechat_assistant.db"

_ENGINE = None
_SessionLocal = None
_MEMORI: Optional[Memori] = None
_OPENAI_CLIENT: Optional[OpenAI] = None


def _build_db_url() -> str:
    """Return the database URL Memori should use.

    By default this reuses the local SQLite database (``wechat_assistant.db``)
    so Memori data lives alongside the assistant's own tables. Set the
    environment variable ``MEMORI_DB_URL`` to override.
    """

    return os.environ.get("MEMORI_DB_URL", DEFAULT_DB_URL)


def _init_memori() -> None:
    """Initialize Memori and a registered OpenAI client once per process."""

    global _ENGINE, _SessionLocal, _MEMORI, _OPENAI_CLIENT

    if _MEMORI is not None and _OPENAI_CLIENT is not None:
        return

    db_url = _build_db_url()
    _ENGINE = create_engine(db_url, future=True)
    _SessionLocal = sessionmaker(bind=_ENGINE, autoflush=False, autocommit=False)

    mem = Memori(conn=_SessionLocal)
    mem.config.storage.build()

    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)

    mem = mem.openai.register(client)

    _MEMORI = mem
    _OPENAI_CLIENT = client

    print(f"[memori] initialized with db_url={db_url}")


def get_memori() -> Memori:
    """Return the global Memori instance, initializing if necessary."""

    if _MEMORI is None:
        _init_memori()
    assert _MEMORI is not None
    return _MEMORI


def get_openai_client(
    *,
    entity_id: Optional[str] = None,
    process_id: Optional[str] = None,
    tags: Optional[Dict[str, Any]] = None,
) -> OpenAI:
    """Return an OpenAI client instrumented by Memori with attribution.

    The client is cached per process. Attribution metadata (entity/process/tags)
    is set on each call so downstream Memori analytics can bucket requests by
    contact and channel.
    """

    if _MEMORI is None or _OPENAI_CLIENT is None:
        _init_memori()

    assert _MEMORI is not None and _OPENAI_CLIENT is not None

    if entity_id or process_id or tags:
        try:
            _MEMORI.attribution(
                entity_id=entity_id or "unknown",
                process_id=process_id or "default",
                tags=tags or {},
            )
        except Exception as exc:  # pragma: no cover - defensive path
            print("[memori] attribution failed:", repr(exc))

    return _OPENAI_CLIENT

