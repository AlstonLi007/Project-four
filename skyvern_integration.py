from __future__ import annotations

"""Minimal Skyvern HTTP client with artifact logging.

This keeps Skyvern usage consistent with the existing SQLite + artifacts
pipeline so browser runs can feed back into LightRAG and downstream analytics.
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

from db import Database
from skyvern_config import SkyvernSettings, get_skyvern_settings


@dataclass
class SkyvernRunResult:
    """Basic representation of a Skyvern run."""

    run_id: str
    status: str
    output_url: Optional[str]
    raw: Dict[str, Any]


class SkyvernClient:
    """Thin Skyvern client using the HTTP API.

    The actual Skyvern endpoints may differ; adjust ``_run_url`` / payload keys
    as needed when you wire this to the live service. All imports are from the
    standard library to keep ``py_compile`` happy even when the Skyvern SDK is
    not installed.
    """

    def __init__(self, settings: Optional[SkyvernSettings] = None) -> None:
        self.settings = settings or get_skyvern_settings()

    def _run_url(self, run_id: Optional[str] = None) -> str:
        base = self.settings.base_url.rstrip("/")
        return f"{base}/v1/runs" if run_id is None else f"{base}/v1/runs/{run_id}"

    @property
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(url, method="POST")
        for key, value in self._headers.items():
            req.add_header(key, value)
        body = json.dumps(payload).encode("utf-8")
        req.data = body
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Skyvern POST failed: {exc} {detail}") from exc

    def _get_json(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
        for key, value in self._headers.items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Skyvern GET failed: {exc} {detail}") from exc

    def create_run(
        self,
        *,
        start_url: str,
        instructions: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SkyvernRunResult:
        payload: Dict[str, Any] = {
            "start_url": start_url,
            "instructions": instructions,
        }
        if metadata:
            payload["metadata"] = metadata

        data = self._post_json(self._run_url(), payload)
        run_id = str(data.get("id") or data.get("run_id") or "unknown")
        status = str(data.get("status") or "unknown")
        output_url = data.get("output_url")
        return SkyvernRunResult(run_id=run_id, status=status, output_url=output_url, raw=data)

    def get_run(self, run_id: str) -> SkyvernRunResult:
        data = self._get_json(self._run_url(run_id))
        status = str(data.get("status") or "unknown")
        output_url = data.get("output_url")
        return SkyvernRunResult(run_id=str(run_id), status=status, output_url=output_url, raw=data)


def log_skyvern_artifact(
    db: Database,
    *,
    kind: str,
    source: str,
    title: str,
    text_content: str,
    meta: Optional[Dict[str, Any]] = None,
    url: Optional[str] = None,
) -> int:
    """Persist a Skyvern run result in the artifacts table.

    Storing the raw result lets LightRAG index it and keeps analytics aware of
    browser automation outputs alongside chat-derived data.
    """

    meta_json = json.dumps(meta or {}, ensure_ascii=False)
    with db._connect() as con:
        cur = con.execute(
            """
            INSERT INTO artifacts (kind, source, url, title, text_content, meta_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (kind, source, url, title, text_content, meta_json),
        )
        return int(cur.lastrowid)
