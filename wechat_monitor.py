from __future__ import annotations

import logging
import os
import signal
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional

# Optional RPA integration: this will be used if rpa_engine.py exists and works.
try:
    from rpa_engine import WeChatRPA, get_default_wechat_rpa  # type: ignore
except Exception:  # rpa_engine missing or broken is OK; we degrade gracefully.
    WeChatRPA = None  # type: ignore[assignment]
    get_default_wechat_rpa = None  # type: ignore[assignment]

log = logging.getLogger("wechat_monitor")

DEFAULT_DB_PATH = os.environ.get("WECHAT_DB_PATH", "wechat_automation.db")
DEFAULT_POLL_INTERVAL = float(os.environ.get("WECHAT_POLL_INTERVAL", "1.0"))


def configure_logging() -> None:
    """Configure root logging once."""
    root = logging.getLogger()
    if root.handlers:
        # Already configured (e.g. by another module)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@dataclass
class WeChatMonitor:
    db_path: str = DEFAULT_DB_PATH
    poll_interval: float = DEFAULT_POLL_INTERVAL
    rpa: "WeChatRPA | None" = None

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self._conn: sqlite3.Connection | None = None

    # === Public API ===

    def start(self) -> None:
        """Start the monitor loop in the foreground (blocking)."""
        configure_logging()
        log.info("Booting WeChat monitor…")
        log.info(
            "Using DB_PATH=%s (can override with WECHAT_DB_PATH env var)",
            self.db_path,
        )
        self._init_db()
        log.info("WeChatMonitor starting in foreground…")
        self._run_loop()

    def request_stop(self) -> None:
        """Ask the loop to shut down cleanly."""
        self._stop_event.set()

    # === Internal helpers ===

    def _init_db(self) -> None:
        """Create a very small DB to log heartbeats."""
        self._conn = sqlite3.connect(self.db_path)
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS heartbeats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc REAL NOT NULL,
                note TEXT
            )
            """
        )
        self._conn.commit()

    def _record_heartbeat(self, note: str) -> None:
        if not self._conn:
            return
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO heartbeats (ts_utc, note) VALUES (?, ?)",
            (time.time(), note),
        )
        self._conn.commit()

    def _run_loop(self) -> None:
        log.info(
            "WeChatMonitor loop starting (poll_interval=%.2fs)…",
            self.poll_interval,
        )
        try:
            while not self._stop_event.is_set():
                self._tick_once()
                time.sleep(self.poll_interval)
        finally:
            log.info("WeChatMonitor loop exiting.")
            if self._conn is not None:
                self._conn.close()

    def _tick_once(self) -> None:
        """
        One heartbeat of the monitor.

        If an RPA engine is available, we let it do a light probe.
        Otherwise we just log our own heartbeat.
        """
        note = "default"

        if self.rpa is not None:
            try:
                self.rpa.heartbeat_probe()
                note = "rpa_ok"
            except Exception as exc:  # noqa: BLE001
                log.warning("RPA heartbeat failed: %s", exc)
                # Fallback to local heartbeat so you still see ticks.
                self._default_tick()
                note = f"rpa_error:{type(exc).__name__}"
        else:
            self._default_tick()

        self._record_heartbeat(note)

    def _default_tick(self) -> None:
        """Local heartbeat when no RPA is wired (or when RPA fails)."""
        log.info("WeChatMonitor heartbeat tick.")


def _install_signal_handlers(monitor: WeChatMonitor) -> None:
    """Handle Ctrl+C / SIGTERM cleanly."""

    def _handler(signum, _frame) -> None:  # type: ignore[override]
        log.info("Received signal %s, requesting shutdown…", signum)
        monitor.request_stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:  # noqa: BLE001
            # Some platforms/signals may not be available – ignore quietly.
            pass


def main() -> None:
    configure_logging()

    # Try to bring up the RPA engine if available.
    rpa: Optional["WeChatRPA"] = None
    if get_default_wechat_rpa is not None:
        try:
            rpa = get_default_wechat_rpa()
            log.info("RPA engine initialized.")
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to initialize RPA engine: %s", exc)
            rpa = None

    monitor = WeChatMonitor(
        db_path=DEFAULT_DB_PATH,
        poll_interval=DEFAULT_POLL_INTERVAL,
        rpa=rpa,
    )
    _install_signal_handlers(monitor)
    monitor.start()


if __name__ == "__main__":
    main()
