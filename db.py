"""SQLite storage layer for WeChat-style conversational logging.

This module keeps contacts, conversations, raw messages, reply candidates,
training examples, and style profiles in a single SQLite database file. It is
framework-light so it can be imported from UI automation code without extra
dependencies.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

DEFAULT_DB_PATH = "wechat_assistant.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS contacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    wechat_id       TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    type            TEXT NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen_at    DATETIME,
    UNIQUE(wechat_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id       INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
    channel          TEXT NOT NULL,
    external_chat_id TEXT,
    created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_message_at  DATETIME
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender          TEXT NOT NULL,
    raw_text        TEXT,
    clean_text      TEXT,
    msg_type        TEXT NOT NULL,
    created_at      DATETIME NOT NULL,
    meta_json       TEXT
);

CREATE TABLE IF NOT EXISTS reply_candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    message_id      INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    candidate_index INTEGER NOT NULL,
    text            TEXT NOT NULL,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    chosen          INTEGER DEFAULT 0,
    chosen_at       DATETIME,
    edited_text     TEXT,
    source          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_examples (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id      INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    context_text    TEXT NOT NULL,
    reply_text      TEXT NOT NULL,
    source          TEXT NOT NULL,
    weight          REAL DEFAULT 1.0,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS style_profiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id      INTEGER NOT NULL REFERENCES contacts(id) ON DELETE CASCADE,
    channel         TEXT NOT NULL,
    label           TEXT NOT NULL,
    prompt_text     TEXT NOT NULL,
    embedding_json  TEXT,
    model_type      TEXT NOT NULL,
    model_id        TEXT,
    version         INTEGER DEFAULT 1,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at      DATETIME,
    UNIQUE(contact_id, channel, label)
);

CREATE TABLE IF NOT EXISTS contact_hparams (
    contact_id      INTEGER PRIMARY KEY REFERENCES contacts(id) ON DELETE CASCADE,
    half_life_days  REAL,
    rag_mode        TEXT NOT NULL DEFAULT 'off'
);

CREATE TABLE IF NOT EXISTS llm_failover_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             DATETIME DEFAULT CURRENT_TIMESTAMP,
    contact_id     INTEGER,
    backend        TEXT,
    error_message  TEXT,
    heuristic_name TEXT
);

CREATE TABLE IF NOT EXISTS artifacts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    kind         TEXT NOT NULL,
    source       TEXT,
    url          TEXT,
    title        TEXT,
    text_content TEXT,
    meta_json    TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Helpful indexes for analytics and retrieval-heavy queries
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created
    ON messages (conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_reply_candidates_message
    ON reply_candidates (message_id);

CREATE INDEX IF NOT EXISTS idx_training_examples_contact_created
    ON training_examples (contact_id, created_at);

CREATE INDEX IF NOT EXISTS idx_artifacts_kind_source_created
    ON artifacts (kind, source, created_at);
"""


@dataclass
class Contact:
    """Simple contact value object for convenience."""

    id: int
    wechat_id: str
    display_name: str
    type: str


class Database:
    """Lightweight SQLite helper with explicit schema management."""

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self.path = Path(path)
        self._ensure_schema()

    @contextmanager
    def _connect(self) -> Iterable[sqlite3.Connection]:
        con = sqlite3.connect(str(self.path))
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)
            self._apply_migrations(con)

    @staticmethod
    def _column_exists(con: sqlite3.Connection, table: str, column: str) -> bool:
        cur = con.execute(f"PRAGMA table_info({table})")
        return any(row[1] == column for row in cur.fetchall())

    def _apply_migrations(self, con: sqlite3.Connection) -> None:
        """Run lightweight, idempotent migrations for schema drift.

        Currently ensures the ``artifacts`` table has a ``url`` column so older
        databases remain compatible with Skyvern artifact logging.
        """

        if self._column_exists(con, "artifacts", "url"):
            return

        try:
            con.execute("ALTER TABLE artifacts ADD COLUMN url TEXT")
        except sqlite3.OperationalError:
            # If the table truly doesn't exist, the main schema creation will
            # create it on the next run; otherwise, fail silently to keep
            # initialization resilient.
            pass

    # -------- contacts & conversations --------

    def get_or_create_contact(
        self, wechat_id: str, display_name: Optional[str] = None, type: str = "private"
    ) -> Contact:
        """Fetch or insert a contact, returning a ``Contact`` dataclass."""

        with self._connect() as con:
            cur = con.execute(
                "SELECT * FROM contacts WHERE wechat_id = ?",
                (wechat_id,),
            )
            row = cur.fetchone()
            if row:
                return Contact(
                    id=row["id"],
                    wechat_id=row["wechat_id"],
                    display_name=row["display_name"],
                    type=row["type"],
                )

            if not display_name:
                display_name = wechat_id

            cur = con.execute(
                """
                INSERT INTO contacts (wechat_id, display_name, type, last_seen_at)
                VALUES (?, ?, ?, ?)
                """,
                (wechat_id, display_name, type, datetime.utcnow().isoformat()),
            )
            cid = cur.lastrowid
            return Contact(id=int(cid), wechat_id=wechat_id, display_name=display_name, type=type)

    def list_contacts(self) -> List[Dict[str, Any]]:
        """Return all contacts with basic metadata."""

        with self._connect() as con:
            cur = con.execute(
                "SELECT id, wechat_id, display_name, type, last_seen_at FROM contacts ORDER BY id ASC"
            )
            return [dict(row) for row in cur.fetchall()]

    def touch_contact(self, contact_id: int) -> None:
        """Update ``last_seen_at`` when a contact is observed in the UI."""

        with self._connect() as con:
            con.execute(
                "UPDATE contacts SET last_seen_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), contact_id),
            )

    def get_or_create_conversation(
        self,
        contact_id: int,
        channel: str = "wechat",
        external_chat_id: Optional[str] = None,
    ) -> int:
        """Return a conversation ID for the given contact + channel."""

        with self._connect() as con:
            cur = con.execute(
                """
                SELECT id FROM conversations
                WHERE contact_id = ? AND channel = ? AND ifnull(external_chat_id, '') = ifnull(?, '')
                """,
                (contact_id, channel, external_chat_id),
            )
            row = cur.fetchone()
            if row:
                return int(row["id"])

            cur = con.execute(
                """
                INSERT INTO conversations (contact_id, channel, external_chat_id, last_message_at)
                VALUES (?, ?, ?, ?)
                """,
                (contact_id, channel, external_chat_id, datetime.utcnow().isoformat()),
            )
            return int(cur.lastrowid)

    # -------- messages & candidates --------

    def log_message(
        self,
        conversation_id: int,
        sender: str,
        raw_text: str,
        msg_type: str = "text",
        created_at: Optional[datetime] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert a message and update the conversation timestamp."""

        if created_at is None:
            created_at = datetime.utcnow()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO messages (conversation_id, sender, raw_text, clean_text,
                                      msg_type, created_at, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    sender,
                    raw_text,
                    raw_text,
                    msg_type,
                    created_at.isoformat(),
                    meta_json,
                ),
            )
            msg_id = int(cur.lastrowid)
            con.execute(
                "UPDATE conversations SET last_message_at = ? WHERE id = ?",
                (created_at.isoformat(), conversation_id),
            )
            return msg_id

    def log_reply_candidates(
        self,
        conversation_id: int,
        message_id: int,
        candidates: List[str],
        source: str = "auto",
    ) -> List[int]:
        """Persist generated reply options for later review/training."""

        ids: List[int] = []
        with self._connect() as con:
            for idx, text in enumerate(candidates, start=1):
                cur = con.execute(
                    """
                    INSERT INTO reply_candidates
                    (conversation_id, message_id, candidate_index, text, source)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (conversation_id, message_id, idx, text, source),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def mark_candidate_chosen(
        self,
        candidate_id: int,
        edited_text: Optional[str] = None,
    ) -> None:
        """Record which candidate was selected (and any edits)."""

        now = datetime.utcnow().isoformat()
        with self._connect() as con:
            if edited_text:
                con.execute(
                    """
                    UPDATE reply_candidates
                    SET chosen = 1, chosen_at = ?, edited_text = ?
                    WHERE id = ?
                    """,
                    (now, edited_text, candidate_id),
                )
            else:
                con.execute(
                    """
                    UPDATE reply_candidates
                    SET chosen = 1, chosen_at = ?
                    WHERE id = ?
                    """,
                    (now, candidate_id),
                )

    # -------- training examples --------

    def add_training_example(
        self,
        contact_id: int,
        conversation_id: int,
        context_text: str,
        reply_text: str,
        source: str = "manual",
        weight: float = 1.0,
    ) -> int:
        """Store a context/reply pair for later fine-tuning."""

        with self._connect() as con:
            cur = con.execute(
                """
                INSERT INTO training_examples
                (contact_id, conversation_id, context_text, reply_text, source, weight)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (contact_id, conversation_id, context_text, reply_text, source, weight),
            )
            return int(cur.lastrowid)

    def get_training_examples(
        self,
        contact_id: int,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch the newest training examples for a contact."""

        with self._connect() as con:
            cur = con.execute(
                """
                SELECT context_text, reply_text, source, weight, created_at, id, conversation_id
                FROM training_examples
                WHERE contact_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (contact_id, limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # -------- helpers for building training context --------

    def get_recent_messages(
        self,
        conversation_id: int,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch recent messages for a conversation in ascending time order."""

        with self._connect() as con:
            cur = con.execute(
                """
                SELECT id, sender, raw_text, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )
            rows = cur.fetchall()

        # Reverse so the caller receives oldest -> newest ordering.
        return list(reversed([dict(r) for r in rows]))

    # -------- style profiles --------

    def get_style_profile(
        self,
        contact_id: int,
        channel: str = "wechat",
        label: str = "default",
    ) -> Optional[Dict[str, Any]]:
        """Return the stored style profile for a contact, if present."""

        with self._connect() as con:
            cur = con.execute(
                """
                SELECT * FROM style_profiles
                WHERE contact_id = ? AND channel = ? AND label = ?
                """,
                (contact_id, channel, label),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def upsert_style_profile(
        self,
        contact_id: int,
        channel: str,
        label: str,
        prompt_text: str,
        model_type: str = "prompt",
        model_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> None:
        """Insert or update a style profile with prompt/model metadata."""

        now = datetime.utcnow().isoformat()
        emb_json = json.dumps(embedding) if embedding is not None else None
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT id, version FROM style_profiles
                WHERE contact_id = ? AND channel = ? AND label = ?
                """,
                (contact_id, channel, label),
            )
            row = cur.fetchone()
            if row:
                current_version = row["version"] or 1
                next_version = int(current_version) + 1
                con.execute(
                    """
                    UPDATE style_profiles
                    SET prompt_text = ?, embedding_json = ?, model_type = ?, model_id = ?,
                        updated_at = ?, version = ?
                    WHERE id = ?
                    """,
                    (prompt_text, emb_json, model_type, model_id, now, next_version, row["id"]),
                )
            else:
                con.execute(
                    """
                    INSERT INTO style_profiles
                    (contact_id, channel, label, prompt_text, embedding_json,
                     model_type, model_id, version, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (contact_id, channel, label, prompt_text, emb_json, model_type, model_id, 1, now, now),
                )

    # -------- contact hyper-parameters --------

    def get_contact_half_life(self, contact_id: int, default: float = 120.0) -> float:
        """Return a contact-specific half-life in days, falling back to ``default``.

        The ``contact_hparams`` table is optional; if no row exists or ``half_life_days``
        is NULL, the provided ``default`` is returned.
        """

        with self._connect() as con:
            cur = con.execute(
                "SELECT half_life_days FROM contact_hparams WHERE contact_id = ?",
                (contact_id,),
            )
            row = cur.fetchone()
        if row is None:
            return float(default)
        value = row["half_life_days"]
        return float(value) if value is not None else float(default)

    def set_contact_half_life(self, contact_id: int, half_life_days: float) -> None:
        """Upsert a per-contact half-life window (in days) for training decay."""

        with self._connect() as con:
            cur = con.execute(
                "SELECT contact_id FROM contact_hparams WHERE contact_id = ?",
                (contact_id,),
            )
            row = cur.fetchone()
            if row:
                con.execute(
                    "UPDATE contact_hparams SET half_life_days = ? WHERE contact_id = ?",
                    (float(half_life_days), contact_id),
                )
            else:
                con.execute(
                    "INSERT INTO contact_hparams (contact_id, half_life_days) VALUES (?, ?)",
                    (contact_id, float(half_life_days)),
                )

    def get_contact_rag_mode(self, contact_id: int, default: str = "off") -> str:
        """Return a per-contact RAG mode ("off"|"light"|"normal"|"aggressive")."""

        with self._connect() as con:
            cur = con.execute(
                "SELECT rag_mode FROM contact_hparams WHERE contact_id = ?",
                (contact_id,),
            )
            row = cur.fetchone()
        if row is None:
            return default
        value = row["rag_mode"]
        return value if value is not None else default

    def set_contact_rag_mode(self, contact_id: int, rag_mode: str) -> None:
        """Upsert a per-contact RAG mode preference."""

        rag_mode_clean = rag_mode or "off"
        with self._connect() as con:
            cur = con.execute(
                "SELECT contact_id FROM contact_hparams WHERE contact_id = ?",
                (contact_id,),
            )
            row = cur.fetchone()
            if row:
                con.execute(
                    "UPDATE contact_hparams SET rag_mode = ? WHERE contact_id = ?",
                    (rag_mode_clean, contact_id),
                )
            else:
                con.execute(
                    "INSERT INTO contact_hparams (contact_id, rag_mode) VALUES (?, ?)",
                    (contact_id, rag_mode_clean),
                )

    def log_llm_failover(
        self,
        contact_id: Optional[int],
        *,
        backend: str,
        error_message: str,
        heuristic_name: str = "_heuristic_replies",
    ) -> None:
        """Record an LLM failure that triggered heuristic fallbacks."""

        with self._connect() as con:
            con.execute(
                """
                INSERT INTO llm_failover_events (contact_id, backend, error_message, heuristic_name)
                VALUES (?, ?, ?, ?)
                """,
                (contact_id, backend, error_message[:500], heuristic_name),
            )
