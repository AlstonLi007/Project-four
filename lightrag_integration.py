from __future__ import annotations

"""LightRAG integration for contact-scoped retrieval and artifacts indexing."""

import asyncio
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lightrag import LightRAG
from lightrag.base import QueryParam
from openai import OpenAI

from db import Database, DEFAULT_DB_PATH
from memori_integration import get_openai_client

RAG_ROOT_DIR = Path("rag_store")
RAG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "text-embedding-3-large"
RAG_LLM_MODEL = "gpt-4.1-mini"


def _memori_client_for_rag(entity_id: str, process_id: str) -> OpenAI:
    """Return a Memori-instrumented OpenAI client for RAG tasks."""

    return get_openai_client(
        entity_id=entity_id,
        process_id=process_id,
        tags={"channel": "wechat", "rag": True},
    )


def _make_embedding_func(entity_id: str):
    async def _embedding(texts: List[str]) -> List[List[float]]:
        def _embed_blocking(batch: List[str]) -> List[List[float]]:
            client = _memori_client_for_rag(entity_id, "wechat_rag_embed")
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            return [record.embedding for record in resp.data]

        return await asyncio.to_thread(_embed_blocking, texts)

    return _embedding


def _make_llm_func(entity_id: str):
    async def _llm(model: str, messages: List[Dict[str, str]], **kwargs) -> str:
        def _completion_blocking() -> str:
            client = _memori_client_for_rag(entity_id, "wechat_rag_query")
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return (resp.choices[0].message.content or "").strip()

        return await asyncio.to_thread(_completion_blocking)

    return _llm


_rag_cache: Dict[str, LightRAG] = {}


def _cache_key(wechat_id: str, working_dir: Path) -> str:
    return f"{working_dir.resolve()}::{wechat_id}"


def get_lightrag_for_contact(wechat_id: str, working_dir: Optional[str] = None) -> LightRAG:
    """Return (and cache) a LightRAG instance scoped to ``wechat_id``."""

    base_dir = Path(working_dir) if working_dir else RAG_ROOT_DIR
    workdir = base_dir / f"wechat_{wechat_id}"
    key = _cache_key(wechat_id, workdir)
    if key in _rag_cache:
        return _rag_cache[key]

    workdir.mkdir(parents=True, exist_ok=True)
    rag = LightRAG(
        working_dir=str(workdir),
        embedding_func=_make_embedding_func(entity_id=f"wechat:{wechat_id}"),
        llm_model_func=_make_llm_func(entity_id=f"wechat:{wechat_id}"),
    )

    _rag_cache[key] = rag
    return rag


async def _index_training_examples_for_contact(
    wechat_id: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 300,
    working_dir: Optional[str] = None,
    style_profile_version: Optional[int] = None,
) -> None:
    """Insert a contact's training examples into their RAG index as a single doc."""

    db = Database(path=db_path)
    contact = db.get_or_create_contact(wechat_id=wechat_id, display_name=wechat_id)
    examples = db.get_training_examples(contact_id=contact.id, limit=limit)
    if not examples:
        return

    blocks: List[str] = []
    for ex in reversed(examples):
        ctx = (ex.get("context_text") or "").strip()
        rep = (ex.get("reply_text") or "").strip()
        if not ctx or not rep:
            continue
        blocks.append(f"【上下文】\n{ctx}\n\n【Alston 回复】\n{rep}\n")

    if not blocks:
        return

    spv = style_profile_version or 0
    doc_id = f"training:{wechat_id}:spv{spv}"
    track_id = f"training_examples:{wechat_id}:spv{spv}"

    metadata = {
        "kind": "training_examples",
        "wechat_id": wechat_id,
        "display_name": contact.display_name,
        "contact_id": contact.id,
        "style_profile_version": spv,
    }

    rag = get_lightrag_for_contact(wechat_id, working_dir=working_dir)
    await rag.ainsert(
        input="\n\n".join(blocks),
        ids=[doc_id],
        track_id=track_id,
        metadatas=[metadata],
    )


def backfill_training_examples_to_rag(
    db_path: str = DEFAULT_DB_PATH,
    *,
    working_dir: str = "rag_store",
    max_examples_per_contact: int = 500,
    style_profile_version: Optional[int] = None,
) -> None:
    """Backfill all contacts' training examples into LightRAG."""

    db = Database(path=db_path)
    contacts = db.list_contacts()

    for contact in contacts:
        asyncio.run(
            _index_training_examples_for_contact(
                contact.get("wechat_id") or str(contact.get("id")),
                db_path=db_path,
                limit=max_examples_per_contact,
                working_dir=working_dir,
                style_profile_version=style_profile_version,
            )
        )


def ensure_index_for_contact(
    wechat_id: str,
    *,
    db_path: str = DEFAULT_DB_PATH,
    limit: int = 300,
    working_dir: str = "rag_store",
    style_profile_version: Optional[int] = None,
) -> None:
    """Synchronously refresh a single contact's RAG index."""

    asyncio.run(
        _index_training_examples_for_contact(
            wechat_id,
            db_path=db_path,
            limit=limit,
            working_dir=working_dir,
            style_profile_version=style_profile_version,
        )
    )


async def _rag_query_for_contact(
    wechat_id: str,
    query: str,
    *,
    top_k: int = 4,
    working_dir: Optional[str] = None,
    style_profile_version: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run a LightRAG query and return a condensed context string + metadata."""

    rag = get_lightrag_for_contact(wechat_id, working_dir=working_dir)
    param = QueryParam(top_k=top_k)
    try:
        result = await rag.aquery_data(query, param=param, model=RAG_LLM_MODEL)
    except Exception as exc:  # pragma: no cover - retrieval errors are non-fatal
        meta = {
            "engine": "LightRAG",
            "model": RAG_LLM_MODEL,
            "mode": getattr(param, "mode", None),
            "top_k": top_k,
            "hit_count": 0,
            "doc_ids": [],
            "working_dir": str(Path(working_dir) if working_dir else RAG_ROOT_DIR),
            "style_profile_version": style_profile_version,
            "wechat_id": wechat_id,
            "error": repr(exc),
        }
        return "", meta

    snippets: List[str] = []
    doc_ids: List[str] = []
    if isinstance(result, dict):
        for key in ("contexts", "docs", "chunks", "sub_queries"):
            value = result.get(key)
            if not value:
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        snippets.append(item)
                    elif isinstance(item, dict):
                        text = (
                            item.get("content")
                            or item.get("text")
                            or item.get("chunk")
                            or ""
                        )
                        if text:
                            snippets.append(text)
                        doc_id = item.get("doc_id") or item.get("id")
                        if doc_id:
                            doc_ids.append(str(doc_id))
    if not snippets:
        context = str(result)[:1000]
    else:
        context = "\n\n---\n\n".join(snippets)
    context = context[:2000]

    meta: Dict[str, Any] = {
        "engine": "LightRAG",
        "model": RAG_LLM_MODEL,
        "mode": getattr(param, "mode", None),
        "top_k": top_k,
        "hit_count": len(snippets),
        "doc_ids": sorted(set(doc_ids)),
        "working_dir": str(Path(working_dir) if working_dir else RAG_ROOT_DIR),
        "style_profile_version": style_profile_version,
        "wechat_id": wechat_id,
    }

    return context, meta


def rag_query_with_meta_for_contact(
    wechat_id: str,
    query: str,
    *,
    top_k: int = 4,
    working_dir: Optional[str] = None,
    style_profile_version: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Synchronous wrapper returning context + metadata."""

    return asyncio.run(
        _rag_query_for_contact(
            wechat_id,
            query=query,
            top_k=top_k,
            working_dir=working_dir,
            style_profile_version=style_profile_version,
        )
    )


def rag_search_for_contact(
    wechat_id: str,
    query: str,
    *,
    top_k: int = 4,
    working_dir: Optional[str] = None,
    style_profile_version: Optional[str] = None,
) -> str:
    """Backward-compatible wrapper returning only the context string."""

    context, _ = rag_query_with_meta_for_contact(
        wechat_id,
        query,
        top_k=top_k,
        working_dir=working_dir,
        style_profile_version=style_profile_version,
    )
    return context


# ---------- Artifacts → LightRAG ----------

def _load_artifacts_grouped(
    db_path: str,
    *,
    kind_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> Dict[Tuple[str, str], List[sqlite3.Row]]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    sql = """
        SELECT id, kind, source, url, title, text_content, meta_json, created_at
        FROM artifacts
    """
    clauses = []
    params: List[object] = []
    if kind_filter:
        clauses.append("kind = ?")
        params.append(kind_filter)
    if source_filter:
        clauses.append("source = ?")
        params.append(source_filter)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY kind, source, datetime(created_at) ASC, id ASC"

    try:
        cur = con.execute(sql, params)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        con.close()
        return {}
    con.close()

    grouped: Dict[Tuple[str, str], List[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], row["source"] or "")] += [row]
    return grouped


def ensure_index_from_artifacts(
    *,
    db_path: str = DEFAULT_DB_PATH,
    working_dir: str = "rag_store",
    kind_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    style_profile_version: Optional[int] = None,
) -> None:
    """Index artifacts (e.g., title crawler output) into LightRAG.

    Each (kind, source) pair becomes a single LightRAG document so you can run
    cross-application retrieval alongside chat context.
    """

    rag = get_lightrag_for_contact("artifacts", working_dir=working_dir)
    groups = _load_artifacts_grouped(db_path, kind_filter=kind_filter, source_filter=source_filter)
    spv = style_profile_version or 0

    for (kind, source), rows in groups.items():
        blocks: List[str] = []
        artifact_ids: List[int] = []
        for row in rows:
            title = (row["title"] or "").strip()
            text_content = (row["text_content"] or "").strip()
            if not text_content:
                continue
            artifact_ids.append(row["id"])
            header = f"# {title}" if title else "# Untitled"
            blocks.append(f"{header}\n{text_content}")

        if not blocks:
            continue

        safe_kind = kind.replace(":", "_")
        safe_source = (source or "unknown").replace(":", "_")
        doc_id = f"artifacts:{safe_kind}:{safe_source}:spv{spv}"
        track_id = f"artifacts:{safe_kind}:{safe_source}:spv{spv}"

        metadata = {
            "kind": kind,
            "source": source,
            "artifact_ids": artifact_ids,
            "style_profile_version": spv,
        }

        rag.insert(
            "\n\n\n".join(blocks),
            ids=[doc_id],
            track_id=track_id,
            metadatas=[metadata],
        )


__all__ = [
    "backfill_training_examples_to_rag",
    "ensure_index_for_contact",
    "ensure_index_from_artifacts",
    "get_lightrag_for_contact",
    "rag_query_with_meta_for_contact",
    "rag_search_for_contact",
]
