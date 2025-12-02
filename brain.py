"""Reply drafting logic powered by the shared SQLite database.

This module wires the UI automation layer to the storage layer. It parses
incoming chat transcripts, records them, retrieves any per-contact style
prompt, and builds an LLM-ready prompt that encourages engaging replies.

External LLM calls are intentionally left as comments so the repository stays
key-free. A lightweight heuristic fallback is returned when no model call is
made.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from db import Database
from nn_infer import StyleNN

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
USE_LOCAL_NN = os.environ.get("USE_LOCAL_NN", "0") == "1"
LOCAL_NN_DIR = os.environ.get("LOCAL_NN_DIR", "nn_models")

GLOBAL_SYSTEM_PROMPT = """
你是 Alston 的私人微信草稿助手，目标：

1. 用最近对话里主要使用的语言回复（如果中文为主就用中文，如果全程英文就用英文），不要中英乱切。
2. 语气：18 岁高中生，真诚、自然、不过度热情，不端着，也不舔。
3. 每条候选回复 1–2 句，尽量控制在 20–50 个汉字或对应长度，不写大段长文。
4. 回复结构优先遵循：
   - 先“接住对方”：简单复述或共情对方说的重点；
   - 然后给一句自己的看法、补充或简单信息；
   - 可以用一句轻量的问题收尾，让对方愿意继续聊（但不要强行尬聊）。
5. 安全与真实：
   - 不要假装自己看过对方没有给出内容的链接、文件、长文，只能表达“有空会看/听起来很有趣”之类的态度；
   - 不要凭空捏造事实、经历、行程、数字或帮对方做过的事；
   - 不要代替 Alston 作出重大承诺（比如借钱、投资、签合同、转发广告、承诺一定帮忙等）。
6. 语气细节：
   - 可以少量使用 emoji，但不要一条消息塞很多；
   - 对长辈/不熟的人稍微更礼貌一点，对同龄朋友可以略微随意。
7. 如果你其实没听懂对方的意思，优先用一句话礼貌追问，把“不确定”说清楚，而不是乱接话。

你返回 2–3 个候选回复，形式为 JSON 列表，例如：["候选 1", "候选 2"]。不要加任何解释文字。
"""


@dataclass
class ParsedMessage:
    sender: str
    text: str
    ts: float
    is_me: bool


def _parse_chat_text(chat_text: str) -> List[ParsedMessage]:
    """Parse newline-delimited chat text into structured messages."""

    messages: List[ParsedMessage] = []
    for idx, raw in enumerate(filter(None, (line.strip() for line in chat_text.splitlines()))):
        sender, text = ("unknown", raw)
        if ":" in raw:
            potential_sender, maybe_text = raw.split(":", 1)
            sender, text = potential_sender.strip() or "unknown", maybe_text.strip()
        is_me = sender.lower() in {"me", "alston", "self", "i"}
        messages.append(ParsedMessage(sender=sender, text=text, ts=float(idx), is_me=is_me))
    return messages


def _build_prompt(style_profile: str, dialogue: List[ParsedMessage]) -> str:
    """Format a prompt for a chat completion model."""

    lines = []
    for msg in dialogue:
        role = "You" if msg.is_me else (msg.sender or "contact")
        lines.append(f"{role}: {msg.text}")
    dialogue_block = "\n".join(lines)
    return f"""
{GLOBAL_SYSTEM_PROMPT}

Here is the contact-specific style guidance (if any):
{style_profile or 'N/A'}

Recent dialogue:
{dialogue_block}

Return 2-3 candidate replies as a JSON list of strings. Each reply should show empathy or invite the other person to continue the conversation while staying natural for Alston.
"""


def _call_llm(prompt: str) -> Optional[List[str]]:
    """Attempt to call OpenAI for real candidates, returning ``None`` on failure."""

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=[{"role": "user", "content": prompt}],
        )
        text = response.output[0].content[0].text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                cleaned = [str(item).strip() for item in parsed if str(item).strip()]
                if cleaned:
                    return cleaned
        except json.JSONDecodeError:
            pass

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines or None
    except Exception as exc:  # pragma: no cover - network dependent
        print("[brain] LLM 调用失败，使用启发式候选:", repr(exc))
        return None


def _heuristic_replies(latest_text: str) -> List[str]:
    """Safe fallback replies when no external model is invoked."""

    suffix = "想多聊聊吗？" if latest_text else "最近怎么样？"
    probes = [
        "再说说你的想法，我在听。",
        "听起来不错，还有别的细节吗？",
        "理解你的感受，有什么我能帮忙的吗？",
    ]
    enriched = f"{latest_text} " if latest_text else ""
    return [
        f"{enriched}我懂你的意思，{suffix}",
        f"{enriched}{probes[0]}",
        f"{enriched}{probes[1]}",
    ]


def draft_reply(
    chat_text: str,
    *,
    wechat_id: str = "unknown",
    display_name: str = "Unknown Contact",
    channel: str = "wechat",
) -> List[str]:
    """Draft reply candidates and persist logs to the shared database."""

    messages = _parse_chat_text(chat_text)
    if not messages:
        return ["你好！最近怎么样？", "嘿，我在呢，有什么想聊的？", "说说你的情况，我帮你看看。"]

    db = Database()
    contact = db.get_or_create_contact(wechat_id=wechat_id, display_name=display_name)
    conversation_id = db.get_or_create_conversation(contact_id=contact.id, channel=channel)

    last_message_id = None
    for msg in messages:
        last_message_id = db.log_message(
            conversation_id=conversation_id,
            sender="self" if msg.is_me else "other",
            raw_text=msg.text,
            msg_type="text",
        )

    profile = db.get_style_profile(contact_id=contact.id, channel=channel, label="default")
    style_prompt = profile.get("prompt_text") if profile else ""

    latest_text = messages[-1].text
    full_context = "\n".join(
        f"{'Me' if msg.is_me else (msg.sender or 'Contact')}: {msg.text}" for msg in messages
    )

    llm_prompt = _build_prompt(style_prompt, messages)

    replies: Optional[List[str]] = None

    if USE_LOCAL_NN:
        nn_model = StyleNN.load(wechat_id, model_dir=LOCAL_NN_DIR)
        if nn_model is not None:
            base_text = full_context or latest_text
            nn_reply = nn_model.generate(base_text)
            if nn_reply and nn_reply.strip():
                replies = [nn_reply.strip()]

    if not replies:
        replies = _call_llm(llm_prompt)

    if not replies:
        replies = _heuristic_replies(latest_text)

    if last_message_id is not None:
        db.log_reply_candidates(
            conversation_id=conversation_id,
            message_id=last_message_id,
            candidates=replies,
            source="auto",
        )

    return replies


def record_sent_reply(
    *,
    wechat_id: str,
    display_name: str,
    channel: str,
    reply_text: str,
    context_window: int = 6,
    quality: str = "normal",
) -> None:
    """Log a sent reply plus its recent context into ``training_examples``.

    Call this immediately after you actually send a message so the system can
    learn from your real wording. It reads the most recent messages in the
    conversation to form the context window and pairs it with the new reply.
    """

    db = Database()
    contact = db.get_or_create_contact(wechat_id=wechat_id, display_name=display_name)
    conversation_id = db.get_or_create_conversation(contact_id=contact.id, channel=channel)

    messages = db.get_recent_messages(conversation_id=conversation_id, limit=context_window + 1)
    if not messages:
        return

    context_messages = messages[:-1] if len(messages) > 1 else messages

    lines = []
    for msg in context_messages:
        role = "我" if msg.get("sender") == "self" else "对方"
        text = (msg.get("raw_text") or "").strip()
        if not text:
            continue
        lines.append(f"{role}: {text}")

    context_text = "\n".join(lines).strip()
    reply_clean = reply_text.strip()

    if not context_text or not reply_clean:
        return

    quality_to_weight = {"normal": 1.0, "star": 3.0, "low": 0.3}
    base_weight = quality_to_weight.get(quality, 1.0)

    source = f"auto-ui:{quality}"

    db.add_training_example(
        contact_id=contact.id,
        conversation_id=conversation_id,
        context_text=context_text,
        reply_text=reply_clean,
        source=source,
        weight=base_weight,
    )


# Optional: simple model skeleton for future offline style imitation.
MODEL_SKELETON = """
import torch
import torch.nn as nn


class SimpleStyleGRU(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embed(src)
        _, hidden = self.encoder(src_emb)
        tgt_emb = self.embed(tgt)
        decoded, _ = self.decoder(tgt_emb, hidden)
        return self.out(decoded)
"""
