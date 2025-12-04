from __future__ import annotations

"""Visual crawler demo that OCRs page titles and stores them in artifacts.

This script demonstrates how to compose :mod:`vision_controller` with a simple
workflow: capture a content region, extract likely titles, store them in the
SQLite artifacts table, and click a "next page" button via template matching.
Configure the constants at the top to fit your target site (templates and
region coordinates)."""

import json
import os
import sqlite3
import time
from typing import List

from ocr_backends import PaddleOcrEngine
from vision_controller import Region, VisionController

DB_PATH = "wechat_assistant.db"

# Update to the rectangle containing the scrollable content list.
CONTENT_REGION = Region(left=100, top=150, width=1200, height=700)

# Template filenames expected inside the ``templates/`` directory.
NEXT_BUTTON_TEMPLATE = "next_button.png"
PAGE_LOADED_MARKER = None  # Optional marker template to confirm page load.


def ensure_artifacts_table(db_path: str) -> None:
    """Create a minimal ``artifacts`` table if it does not exist."""

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            kind         TEXT NOT NULL,
            source       TEXT,
            url          TEXT,
            title        TEXT,
            text_content TEXT,
            meta_json    TEXT,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.commit()
    con.close()


def extract_titles(raw_text: str) -> List[str]:
    """Heuristic line filtering to pick probable titles from OCR output."""

    lines = [line.strip() for line in raw_text.splitlines()]
    candidates = [line for line in lines if 10 <= len(line) <= 80]

    seen: set[str] = set()
    titles: List[str] = []
    for line in candidates:
        if line in seen:
            continue
        seen.add(line)
        titles.append(line)
    return titles


def save_titles_to_db(db_path: str, titles: List[str], page_index: int, url: str | None = None) -> None:
    """Persist extracted titles to the ``artifacts`` table."""

    ensure_artifacts_table(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    meta = {
        "page_index": page_index,
        "url": url,
        "title_count": len(titles),
        "source": "visual_crawler_demo",
    }
    text_content = "\n".join(titles)
    title = f"visual_page_{page_index}"

    cur.execute(
        """
        INSERT INTO artifacts (kind, source, url, title, text_content, meta_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "page_titles",
            "visual_crawler_demo",
            url,
            title,
            text_content,
            json.dumps(meta, ensure_ascii=False),
        ),
    )
    con.commit()
    con.close()
    print(f"[db] saved page {page_index} with {len(titles)} titles")


def wait_for_page_loaded(vc: VisionController, timeout: float = 10.0) -> bool:
    """Wait until a marker appears or a simple delay elapses."""

    if PAGE_LOADED_MARKER is None:
        time.sleep(2.0)
        return True

    start = time.time()
    while time.time() - start < timeout:
        located = vc.locate_template(PAGE_LOADED_MARKER, threshold=0.8)
        if located:
            return True
        time.sleep(0.5)
    return False


def main() -> None:
    use_paddle = bool(int(os.getenv("USE_PADDLE_OCR", "0")))
    ocr_engine = PaddleOcrEngine(lang="ch") if use_paddle else None

    vc = VisionController(
        template_dir="templates",
        ocr_lang="eng",
        debug=True,
        ocr_engine=ocr_engine,
    )

    max_pages = 5
    current_url = None  # Visual demo only; add browser integrations if needed.

    print("[demo] Open the target list page, then press Enter to start.")
    input()

    for page_idx in range(1, max_pages + 1):
        print(f"\n[demo] ===== Page {page_idx} =====")

        if not wait_for_page_loaded(vc):
            print("[demo] page load timeout, stop.")
            break

        raw_text = vc.read_region(CONTENT_REGION, lang=None, preprocess=True)
        titles = extract_titles(raw_text)

        if titles:
            for t in titles:
                print("  -", t)
            save_titles_to_db(DB_PATH, titles, page_idx, url=current_url)
        else:
            print("[demo] no titles detected on this page (adjust OCR region/thresholds)")

        print("[demo] clicking next page...")
        ok = vc.click_template(NEXT_BUTTON_TEMPLATE, threshold=0.85)
        if not ok:
            print("[demo] next button not found, stop.")
            break

        time.sleep(2.0)

    print("\n[demo] done.")


if __name__ == "__main__":
    main()
