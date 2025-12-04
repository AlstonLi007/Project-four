from __future__ import annotations

"""Inference helper to load per-contact GRU models and generate replies."""

from pathlib import Path
from typing import Optional

import torch

from nn_model import Seq2SeqGRU
from nn_vocab import Vocab


class StyleNN:
    """Wrapper around a trained GRU style model and its vocabulary."""

    def __init__(self, model: Seq2SeqGRU, vocab: Vocab, device: torch.device) -> None:
        self.model = model
        self.vocab = vocab
        self.device = device

    @classmethod
    def load(cls, wechat_id: str, model_dir: str = "nn_models") -> Optional["StyleNN"]:
        """Load model weights and vocabulary for a given contact if present."""

        model_path = Path(model_dir) / f"{wechat_id}_seq2seq.pt"
        vocab_path = Path(model_dir) / f"{wechat_id}_vocab.json"
        if not model_path.exists() or not vocab_path.exists():
            return None

        vocab = Vocab.load(vocab_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Seq2SeqGRU(vocab_size=len(vocab.itos)).to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return cls(model=model, vocab=vocab, device=device)

    def generate(self, context_text: str, max_len: int = 80) -> str:
        """Greedy-decode a reply from the trained model."""

        self.model.eval()
        with torch.no_grad():
            src_ids = self.vocab.encode(context_text, add_sos_eos=False)
            src = torch.tensor(src_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            hidden = self.model.encode(src)
            cur = torch.tensor([self.vocab.sos_id], dtype=torch.long, device=self.device)
            outputs: list[int] = []

            for _ in range(max_len):
                logits, hidden = self.model.decode_step(cur, hidden)
                next_id = int(torch.argmax(logits, dim=-1)[0].item())
                if next_id == self.vocab.eos_id:
                    break
                outputs.append(next_id)
                cur = torch.tensor([next_id], dtype=torch.long, device=self.device)

            return self.vocab.decode(outputs, strip_special=True)
