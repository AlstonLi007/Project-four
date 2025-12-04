from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    import uiautomation as auto
except ImportError:  # pragma: no cover - runtime dependency
    auto = None  # type: ignore[assignment]

log = logging.getLogger("rpa_engine")


@dataclass
class WeChatRPA:
    """
    Very small RPA wrapper around the WeChat desktop window.

    For now this is intentionally conservative:
    - We only locate the window and bring it to the front.
    - We log a heartbeat that the window is alive.
    - Hooks for reading/sending messages are stubbed for later.
    """

    wechat_window_name: str = "微信"  # Chinese title is common
    wechat_class_name: str = "WeChatMainWndForPC"
    find_timeout: float = 5.0  # seconds to wait when searching for the window

    def __post_init__(self) -> None:
        if auto is None:
            raise ImportError(
                "uiautomation is not installed. "
                "Run 'pip install uiautomation' inside your virtualenv."
            )
        self._window: Optional[Any] = None

    # === Core window discovery ===

    def _locate_window(self) -> Any:
        """
        Find and cache the WeChat main window.

        Raises RuntimeError if WeChat is not running / not found.
        """
        # If we already cached a window, try to reuse it.
        if self._window is not None:
            try:
                # Quick existence check – 0-second timeout: do not block.
                if self._window.Exists(0, 0):
                    return self._window
            except Exception:  # noqa: BLE001
                self._window = None

        win = self._try_get_window()
        if win is None:
            raise RuntimeError(
                "WeChat window not found. "
                "Make sure the WeChat desktop app is running and logged in."
            )

        self._window = win
        return win

    def _try_get_window(self) -> Optional[Any]:
        """
        Try a couple of common search strategies for WeChat.
        Returns a WindowControl or None.
        """
        if auto is None:
            return None

        specs: List[Dict[str, Any]] = []
        if self.wechat_class_name:
            specs.append({"ClassName": self.wechat_class_name})
        if self.wechat_window_name:
            specs.append({"Name": self.wechat_window_name})

        # 1) Try direct spec-based search (class name, then window name)
        for spec in specs:
            try:
                win = auto.WindowControl(searchDepth=1, **spec)
                # Block for up to find_timeout seconds to see if it appears.
                if win.Exists(self.find_timeout, 0.5):
                    log.info("Found WeChat window with spec %r.", spec)
                    return win
            except Exception as exc:  # noqa: BLE001
                log.debug("Window search with spec %r failed: %s", spec, exc)

        # 2) Fuzzy fallback: scan all top-level windows and look for 微信 / WeChat
        try:
            root = auto.GetRootControl()
            for child in root.GetChildren():
                try:
                    name = getattr(child, "Name", None)
                    cls = getattr(child, "ClassName", None)
                except Exception:  # noqa: BLE001
                    continue

                if not name:
                    continue

                if "微信" in str(name) or "WeChat" in str(name):
                    log.info(
                        "Found WeChat window by fuzzy match (Name=%r, ClassName=%r).",
                        name,
                        cls,
                    )
                    return child
        except Exception as exc:  # noqa: BLE001
            log.debug("Fuzzy top-level window search failed: %s", exc)

        return None

    # === Public hooks used by wechat_monitor ===

    def heartbeat_probe(self) -> None:
        """
        Lightweight check used by wechat_monitor.

        - Ensures the WeChat window exists.
        - Tries to bring it to the foreground.
        - Logs a clear INFO-level message.
        """
        win = self._locate_window()

        try:
            win.SetActive()
        except Exception as exc:  # noqa: BLE001
            log.debug("SetActive on WeChat window failed: %s", exc)

        name = getattr(win, "Name", None)
        cls = getattr(win, "ClassName", None)
        log.info(
            "RPA heartbeat: WeChat window alive (Name=%r, ClassName=%r).",
            name,
            cls,
        )

    # === Future APIs: message reading/sending ===

    def get_recent_messages(self, max_chats: int = 5) -> list[dict]:
        """
        Placeholder for later.

        Eventually:
        - Inspect the chat list
        - Return structured recent messages for the brain.
        """
        log.debug(
            "get_recent_messages(max_chats=%s) called; not implemented yet.",
            max_chats,
        )
        return []

    def send_text_reply(self, chat_identifier: str, text: str) -> None:
        """
        Placeholder for later.

        Eventually:
        - Focus the chat by name / id
        - Send the given text.
        """
        log.debug(
            "send_text_reply(chat_identifier=%r, text=%r) called; "
            "not implemented yet.",
            chat_identifier,
            text,
        )


def get_default_wechat_rpa() -> WeChatRPA:
    """
    Factory used by wechat_monitor to obtain a default engine.
    """
    return WeChatRPA()
