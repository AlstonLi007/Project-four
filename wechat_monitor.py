"""Integration harness for WeChat UI automation.

The real UI automation logic is outside this repository. This module shows how
to connect scraped chat transcripts to ``brain.draft_reply`` so replies and logs
flow into the shared SQLite database.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import brain


class WeChatMonitor:
    """Minimal harness illustrating integration points."""

    def __init__(self, platform_id: str = "unknown") -> None:
        self.platform_id = platform_id

    def process_chat_context(
        self,
        chat_text: str,
        display_name: str,
        *,
        wechat_id: str | None = None,
    ) -> list[str]:
        """Process a single chat transcript and return candidate replies.

        Passing both ``wechat_id`` and ``display_name`` lets the database keep
        a stable contact key (ID) while still storing a friendly label, which
        mirrors how the real ``wechat_global_monitor.py`` should call
        ``brain.draft_reply``.
        """

        wechat_id = wechat_id or display_name
        return brain.draft_reply(
            chat_text,
            wechat_id=wechat_id,
            display_name=display_name,
            channel="wechat",
        )

    def run_forever(self, contexts: Iterable[Sequence[str]]) -> None:
        """Stub loop that simulates the polling behavior.

        Args:
            contexts: Iterable of tuples describing chats. Either
                ``(display_name, chat_text)`` or ``(wechat_id, display_name,
                chat_text)``.
        """

        for entry in contexts:
            if len(entry) == 2:
                display_name, chat_text = entry
                wechat_id = display_name
            elif len(entry) == 3:
                wechat_id, display_name, chat_text = entry
            else:  # pragma: no cover - demonstration guard
                raise ValueError("Context tuples must be (display_name, chat_text) or (wechat_id, display_name, chat_text)")

            replies = self.process_chat_context(chat_text, display_name, wechat_id=wechat_id)
            self._send_to_control_chat(display_name, replies)

    def _send_to_control_chat(self, display_name: str, replies: list[str]) -> None:
        """Placeholder for routing candidates to the control chat."""

        print(f"--- {display_name} ---")
        for reply in replies:
            print(f"Candidate: {reply}")


def demo() -> None:
    """Simple offline demonstration."""

    monitor = WeChatMonitor(platform_id="demo")
    sample_contexts = [
        (
            "helper-contact-id",
            "文件传输助手",
            """Me: Hey, how's your day?\nFriend: Pretty good, thinking about a weekend trip.""",
        )
    ]
    monitor.run_forever(sample_contexts)


if __name__ == "__main__":
    demo()
