"""Utilities for synthesizing per-contact style prompts from chat history.

The functions here are intended to be run offline or via a cron job. They pull
recent training examples from the SQLite database, ask an LLM to summarize the
tone and structure, and write that prompt back to ``style_profiles``.
"""
from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from db import Database

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")


STYLE_SYSTEM_PROMPT = """
你是一个“聊天风格分析助手”。

现在给你若干段微信聊天样本，主角都是同一个人 Alston 和同一个联系人之间的对话。
你的任务是：总结出 Alston 针对“这个联系人”时的说话习惯，以便后续作为提示词给另一个模型模仿。

请在 300 字以内，用要点式说明：
- 语气（正式/随意、有没有玩笑）
- 长度偏好（通常是短句还是长段）
- Emoji 使用情况
- 对对方的称呼习惯
- 避免说的话/雷区（如果样本中能看出）
- 适合的回复结构模板（例如：先回应情绪，再给一点信息，最后留个尾巴）

不要复述原文内容，只总结风格。
"""


def build_style_prompt_for_contact(
    db: Database,
    contact_wechat_id: str,
    display_name: Optional[str] = None,
    max_examples: int = 50,
) -> Optional[str]:
    """Generate and store a style prompt for the given contact.

    Args:
        db: Database instance.
        contact_wechat_id: Contact key as used in WeChat.
        display_name: Friendly name to fall back to when creating the contact.
        max_examples: How many most-recent examples to feed into the summarizer.
    Returns:
        The summarized prompt text, or ``None`` if no examples exist.
    """

    contact = db.get_or_create_contact(
        wechat_id=contact_wechat_id,
        display_name=display_name or contact_wechat_id,
    )
    examples = db.get_training_examples(contact_id=contact.id, limit=max_examples)
    if not examples:
        print(f"[style] contact {contact_wechat_id} has no training examples; skip")
        return None

    chunks = []
    for ex in examples:
        ctx = ex["context_text"].strip()
        rep = ex["reply_text"].strip()
        chunks.append(f"【上下文】\n{ctx}\n【Alston 回复】\n{rep}\n")
    payload = "\n\n".join(chunks)

    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": STYLE_SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
    )

    prompt_text = response.output[0].content[0].text.strip()
    db.upsert_style_profile(
        contact_id=contact.id,
        channel="wechat",
        label="default",
        prompt_text=prompt_text,
        model_type="prompt",
        model_id=None,
    )

    print(f"[style] contact {contact_wechat_id} style profile updated")
    return prompt_text


if __name__ == "__main__":
    database = Database()
    # Example run for a specific contact or group.
    build_style_prompt_for_contact(database, contact_wechat_id="妈妈", display_name="妈妈")
