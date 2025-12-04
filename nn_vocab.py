from __future__ import annotations

"""Character-level vocabulary utilities for the local style model."""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


class Vocab:
    """Minimal character vocabulary with serialization helpers."""

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos
        self.pad_id = self.stoi["<pad>"]
        self.sos_id = self.stoi["<s>"]
        self.eos_id = self.stoi["</s>"]
        self.unk_id = self.stoi["<unk>"]

    @classmethod
    def build(cls, texts: Iterable[str], max_size: int = 2000, min_freq: int = 1) -> "Vocab":
        """Build a vocabulary from raw text samples."""

        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(list(text))

        itos: List[str] = SPECIAL_TOKENS.copy()
        for ch, freq in counter.most_common():
            if freq < min_freq:
                continue
            if ch in itos:
                continue
            itos.append(ch)
            if len(itos) >= max_size:
                break

        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)

    def encode(self, text: str, add_sos_eos: bool = True) -> List[int]:
        """Convert text to token IDs, optionally wrapping with SOS/EOS."""

        ids = [self.stoi.get(ch, self.unk_id) for ch in text]
        if add_sos_eos:
            return [self.sos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Turn token IDs back into text, optionally removing special tokens."""

        chars: List[str] = []
        for token_id in ids:
            if token_id < 0 or token_id >= len(self.itos):
                continue
            ch = self.itos[token_id]
            if strip_special and ch in SPECIAL_TOKENS:
                continue
            chars.append(ch)
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        """Persist the vocabulary to disk as JSON."""

        data = {"itos": self.itos}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        """Load a vocabulary from disk."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        itos = data["itos"]
        stoi = {ch: i for i, ch in enumerate(itos)}
        return cls(stoi=stoi, itos=itos)
