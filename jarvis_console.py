#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Console entrypoint for the Jarvis WeChat assistant.

This script initializes logging, loads environment configuration, attaches to the
WeChat desktop client (via the vision/RPA helpers), and enters a loop that:

1) polls for new messages,
2) generates reply candidates with ``brain.draft_reply``,
3) sends the first candidate back through RPA,
4) logs the sent reply via ``record_sent_reply`` for continual learning.

It is intentionally lightweight so it can be packaged into a single executable
(e.g., with PyInstaller) while reusing the existing database, RAG, and Memori
instrumentation implemented elsewhere in the repository.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# Optional .env loader; if unavailable, the app still runs.
try:  # pragma: no cover - convenience dependency
    from dotenv import load_dotenv  # type: ignore

    _HAS_DOTENV = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_DOTENV = False

from brain import draft_reply, record_sent_reply
from db import Database
from rpa_engine import ScreenCapture, TemplateMatcher, VisionController

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & logging helpers
# ---------------------------------------------------------------------------


def load_env() -> None:
    """Load environment variables from a ``.env`` file if present.

    The search path mirrors typical developer workflows:
    - If running from a PyInstaller-built executable, look beside the binary.
    - Otherwise, fall back to ``.env`` in the current working directory.
    """

    if _HAS_DOTENV:
        base_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path.cwd()
        env_path = base_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=str(env_path), override=False)
            LOGGER.info("Loaded environment variables from %s", env_path)


def setup_logging(log_dir: Optional[Path] = None) -> None:
    """Configure console logging and optional file logging."""

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt)

    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "jarvis.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(file_handler)
        LOGGER.info("File logging enabled: %s", log_dir / "jarvis.log")


# ---------------------------------------------------------------------------
# Message & controller abstractions
# ---------------------------------------------------------------------------


@dataclass
class Message:
    """Uniform message representation passed through the pipeline."""

    chat_id: str
    chat_name: str
    sender_name: str
    text: str
    is_group: bool = False
    timestamp: float = 0.0
    raw: str = ""


class WeChatController:
    """Minimal WeChat UI automation wrapper.

    This is intentionally conservative: it sets up the vision helpers but leaves
    ``fetch_new_messages`` and ``send_text`` as TODOs for your environment. Once
    these are implemented with the templates/regions that match your WeChat UI,
    the rest of the pipeline (DB, RAG, Memori, analytics) works unchanged.
    """

    def __init__(self, *, debug: bool = False) -> None:
        self.debug = debug
        self.matcher = TemplateMatcher(templates=[])
        self.capture = ScreenCapture()
        self.vision = VisionController(self.matcher, self.capture)

    def attach(self) -> None:
        """Attach to the WeChat window.

        Implement this using pyautogui/window handles or template matching to
        bring the WeChat window to the foreground. For now it is a stub.
        """

        LOGGER.info("Attempting to locate WeChat window (stub).")

    def fetch_new_messages(self) -> List[Message]:
        """Poll for unread messages.

        Replace the stub with OCR/templating logic that reads the message list
        and returns new messages as ``Message`` objects. Returning an empty list
        simply keeps the loop alive.
        """

        return []

    def send_text(self, chat_id: str, text: str) -> None:
        """Send a text reply into the active chat window.

        Use the vision controller to focus the input box, paste/type the text,
        and click the send button. Left as a stub so the executable can be
        built before UI offsets are finalized.
        """

        LOGGER.info("[SEND] chat=%s text=%s", chat_id, text)

    def dispose(self) -> None:
        """Clean up resources (placeholder for symmetry)."""

        LOGGER.info("WeChat controller disposed.")


# ---------------------------------------------------------------------------
# Main assistant loop
# ---------------------------------------------------------------------------


class JarvisAssistant:
    """Drives WeChat polling, reply drafting, and logging."""

    def __init__(self) -> None:
        self.db = Database()
        self.controller = WeChatController(debug=os.getenv("DEBUG_WECHAT", "0") == "1")
        self.running = True

    def start(self) -> None:
        LOGGER.info("🚀 Jarvis assistant starting up...")
        self.controller.attach()

        signal.signal(signal.SIGINT, self._handle_exit)
        signal.signal(signal.SIGTERM, self._handle_exit)

        try:
            while self.running:
                messages = self.controller.fetch_new_messages()
                if not messages:
                    time.sleep(1.0)
                    continue

                for msg in messages:
                    self.process_message(msg)
        finally:
            self.shutdown()

    def process_message(self, msg: Message) -> None:
        LOGGER.info("[NEW] chat=%s sender=%s text=%s", msg.chat_name, msg.sender_name, msg.text)

        candidates = draft_reply(
            msg.text,
            wechat_id=msg.chat_id or msg.chat_name,
            display_name=msg.chat_name or "Unknown",
            channel="wechat",
        )

        if not candidates:
            LOGGER.warning("No reply candidates generated; skipping send.")
            return

        reply = candidates[0]
        self.controller.send_text(msg.chat_id, reply)

        record_sent_reply(
            wechat_id=msg.chat_id or msg.chat_name,
            display_name=msg.chat_name or "Unknown",
            channel="wechat",
            reply_text=reply,
            context_window=6,
            quality="normal",
        )

    def _handle_exit(self, signum, frame) -> None:  # pragma: no cover - signal handler
        LOGGER.info("Received signal %s; shutting down...", signum)
        self.running = False

    def shutdown(self) -> None:
        LOGGER.info("Cleaning up controller...")
        self.controller.dispose()
        LOGGER.info("Jarvis assistant stopped.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    load_env()

    log_to_file = os.getenv("JARVIS_LOG_TO_FILE", "1") == "1"
    log_dir = Path("logs") if log_to_file else None
    setup_logging(log_dir)

    assistant = JarvisAssistant()
    assistant.start()


if __name__ == "__main__":
    main()
