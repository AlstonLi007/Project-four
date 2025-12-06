from __future__ import annotations

"""
Filesystem indexer: scan local directories → artifacts table.

- kind = "file"
- source = "local_fs"
- url = file://<absolute_path>
- title = filename
- text_content = extracted plain text
- meta_json = {
    "path": "...",
    "size_bytes": ...,
    "mtime": "...",
    "sha256": "...",
    "ext": ".pdf",
    "tags": [...],
  }

Usage examples:

    # 简单扫描两个目录
    python fs_indexer.py scan --root "D:/School" --root "C:/Projects" --db wechat_assistant.db

Later, run:

    python backfill_rag_from_db.py --include-artifacts --artifacts-kind file --artifacts-source local_fs
"""

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from db import DEFAULT_DB_PATH

# -------- text extractors --------


def _read_text_file(path: Path, max_bytes: int) -> str:
    data = path.read_bytes()[:max_bytes]
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _read_code_file(path: Path, max_bytes: int) -> str:
    return _read_text_file(path, max_bytes)


def _read_docx(path: Path, max_bytes: int) -> str:
    try:
        import docx  # type: ignore
    except ImportError:
        return ""

    try:
        doc = docx.Document(str(path))
        parts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)
        return "\n".join(parts)[: max_bytes // 2]
    except Exception:
        return ""


def _read_pdf(path: Path, max_bytes: int) -> str:
    try:
        import PyPDF2  # type: ignore
    except ImportError:
        return ""

    try:
        text_parts: List[str] = []
        with path.open("rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception:
                    page_text = ""
                if page_text:
                    text_parts.append(page_text)
                if sum(len(p) for p in text_parts) > max_bytes:
                    break
        return "\n\n".join(text_parts)[:max_bytes]
    except Exception:
        return ""


def extract_text(path: Path, max_bytes: int = 200_000) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md", ".rst"}:
        return _read_text_file(path, max_bytes)
    if ext in {".py", ".ipynb", ".js", ".ts", ".java", ".c", ".cpp", ".cs"}:
        return _read_code_file(path, max_bytes)
    if ext in {".docx"}:
        return _read_docx(path, max_bytes)
    if ext in {".pdf"}:
        return _read_pdf(path, max_bytes)
    return _read_text_file(path, max_bytes)


# -------- hashing & meta --------


def file_sha256(path: Path, max_bytes: int = 5_000_000) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65_536)
            if not chunk:
                break
            h.update(chunk)
            max_bytes -= len(chunk)
            if max_bytes <= 0:
                break
    return h.hexdigest()


def file_meta(path: Path) -> Tuple[int, str]:
    st = path.stat()
    size = st.st_size
    mtime = datetime.fromtimestamp(st.st_mtime).isoformat()
    return size, mtime


# -------- artifacts I/O --------


def _open_db(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con


def _find_latest_artifact(
    con: sqlite3.Connection,
    *,
    kind: str,
    source: str,
    url: str,
) -> Optional[sqlite3.Row]:
    cur = con.execute(
        """
        SELECT id, meta_json, created_at
        FROM artifacts
        WHERE kind = ? AND source = ? AND url = ?
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT 1
        """,
        (kind, source, url),
    )
    return cur.fetchone()


def _insert_artifact(
    con: sqlite3.Connection,
    *,
    kind: str,
    source: str,
    url: str,
    title: str,
    text_content: str,
    meta: dict,
) -> int:
    cur = con.execute(
        """
        INSERT INTO artifacts (kind, source, url, title, text_content, meta_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (kind, source, url, title, text_content, json.dumps(meta, ensure_ascii=False)),
    )
    return int(cur.lastrowid)


# -------- scanning --------

DEFAULT_IGNORES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".idea",
    ".vscode",
}

BINARY_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".heic",
    ".mov",
    ".mp4",
    ".avi",
    ".mkv",
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
}


def iter_files(
    roots: Iterable[Path],
    *,
    follow_symlinks: bool = False,
) -> Iterable[Path]:
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORES and not d.startswith(".")]
            base = Path(dirpath)
            for name in filenames:
                p = (base / name).resolve()
                if p in seen:
                    continue
                seen.add(p)
                if p.suffix.lower() in BINARY_EXTS:
                    continue
                yield p


def scan_once(
    *,
    db_path: str,
    roots: List[str],
    max_bytes_per_file: int = 200_000,
    max_files: Optional[int] = None,
    dry_run: bool = False,
) -> None:
    if not roots:
        print("[fs_indexer] no roots provided, nothing to do.")
        return

    root_paths = [Path(r).expanduser() for r in roots]
    con = _open_db(db_path)
    try:
        count = 0
        inserted = 0
        skipped_unchanged = 0
        for path in iter_files(root_paths):
            count += 1
            if max_files is not None and count > max_files:
                break

            abs_path = str(path.resolve())
            url = "file://" + abs_path.replace("\\", "/")
            size_bytes, mtime = file_meta(path)
            sha = file_sha256(path)

            existing = _find_latest_artifact(con, kind="file", source="local_fs", url=url)
            if existing is not None:
                try:
                    meta = json.loads(existing["meta_json"] or "{}")
                except Exception:
                    meta = {}
                if meta.get("sha256") == sha:
                    skipped_unchanged += 1
                    continue

            text = extract_text(path, max_bytes=max_bytes_per_file).strip()
            if not text:
                continue

            meta = {
                "path": abs_path,
                "size_bytes": size_bytes,
                "mtime": mtime,
                "sha256": sha,
                "ext": path.suffix.lower(),
                "tags": ["channel:fs", "source:local_fs"],
            }

            title = path.name
            if dry_run:
                print(f"[dry-run] would index: {abs_path}")
            else:
                _insert_artifact(
                    con,
                    kind="file",
                    source="local_fs",
                    url=url,
                    title=title,
                    text_content=text,
                    meta=meta,
                )
                inserted += 1

        if not dry_run:
            con.commit()

        print(
            f"[fs_indexer] scanned={count}, inserted={inserted}, skipped_unchanged={skipped_unchanged}, db={db_path}"
        )
    finally:
        con.close()


# -------- CLI --------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan local directories and index files into artifacts(kind='file', source='local_fs')."
    )
    parser.add_argument(
        "command",
        choices=["scan"],
        help="Only 'scan' is supported for now.",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite DB (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Root directory to scan (can be repeated).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of files to process.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=200_000,
        help="Max bytes of text to keep per file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be indexed without writing to DB.",
    )

    args = parser.parse_args()

    if args.command == "scan":
        scan_once(
            db_path=args.db,
            roots=args.root,
            max_bytes_per_file=args.max_bytes,
            max_files=args.max_files,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
