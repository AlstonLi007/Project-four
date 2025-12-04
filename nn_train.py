from __future__ import annotations

"""Training script to fit a per-contact GRU seq2seq style model."""

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from db import Database
from nn_model import Seq2SeqGRU
from nn_vocab import Vocab


def _time_decay_weight(age_days: float, *, half_life_days: float) -> float:
    """Compute an exponential time-decay factor based on example age.

    ``half_life_days`` controls how quickly older examples lose influence (120 days ≈ 4 months).
    The half-life is clamped to a tiny positive value to avoid division by zero
    if misconfigured.
    """

    if age_days <= 0:
        return 1.0

    safe_half_life = max(float(half_life_days), 1e-3)
    lam = math.log(2.0) / safe_half_life
    return math.exp(-lam * float(age_days))


class StyleDataset(Dataset):
    """Dataset wrapping context/reply text pairs with per-example weights."""

    def __init__(self, pairs: List[Tuple[str, str, float]], vocab: Vocab, max_len: int = 200) -> None:
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:  # pragma: no cover - simple container
        return len(self.pairs)

    def _truncate(self, ids: List[int]) -> List[int]:
        if len(ids) > self.max_len:
            return ids[: self.max_len]
        return ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ctx, rep, w = self.pairs[idx]
        src_ids = self._truncate(self.vocab.encode(ctx, add_sos_eos=False))
        tgt_ids = self._truncate(self.vocab.encode(rep, add_sos_eos=True))
        src = torch.tensor(src_ids, dtype=torch.long)
        tgt = torch.tensor(tgt_ids, dtype=torch.long)
        weight = torch.tensor(float(w), dtype=torch.float32)
        return src, tgt, weight


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], pad_id: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences for batching."""

    src_seqs, tgt_seqs, weights = zip(*batch)
    src_lens = [len(seq) for seq in src_seqs]
    tgt_lens = [len(seq) for seq in tgt_seqs]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    batch_size = len(batch)
    src_padded = torch.full((batch_size, max_src), pad_id, dtype=torch.long)
    tgt_padded = torch.full((batch_size, max_tgt), pad_id, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        src_padded[i, : len(src)] = src
        tgt_padded[i, : len(tgt)] = tgt
    weight_tensor = torch.stack(weights)
    return src_padded, tgt_padded, weight_tensor


def load_pairs_for_contact(
    db: Database,
    wechat_id: str,
    *,
    limit: int = 200,
    default_half_life_days: float = 120.0,
) -> Tuple[List[Tuple[str, str, float]], float]:
    """Load context/reply pairs + time-decayed weights for a contact.

    Returns both the weighted pairs and the resolved half-life (in days) used
    for decay so callers can surface debug information.
    """

    contact = db.get_or_create_contact(wechat_id=wechat_id, display_name=wechat_id)
    half_life_days = db.get_contact_half_life(contact_id=contact.id, default=default_half_life_days)

    examples = db.get_training_examples(contact_id=contact.id, limit=limit)
    pairs: List[Tuple[str, str, float]] = []

    now = datetime.utcnow()

    for ex in examples:
        ctx = (ex.get("context_text") or "").strip()
        rep = (ex.get("reply_text") or "").strip()
        if not ctx or not rep:
            continue

        base_w = float(ex.get("weight", 1.0) or 1.0)

        created_str = ex.get("created_at")
        try:
            created_at = datetime.fromisoformat(created_str) if created_str else now
        except Exception:
            created_at = now

        age_days = max((now - created_at).days, 0)
        decay = _time_decay_weight(age_days, half_life_days=half_life_days)
        final_w = base_w * decay

        pairs.append((ctx, rep, final_w))

    return pairs, half_life_days


def train_for_contact(
    db_path: str,
    wechat_id: str,
    out_dir: str = "nn_models",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    default_half_life_days: float = 120.0,
) -> None:
    """Train a per-contact GRU model using stored training examples."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    db = Database(path=db_path)
    pairs, half_life_days = load_pairs_for_contact(
        db,
        wechat_id=wechat_id,
        limit=500,
        default_half_life_days=default_half_life_days,
    )
    if len(pairs) < 10:
        raise RuntimeError(f"样本太少（{len(pairs)} 条），先多积累一点 training_examples 再训。")

    weights = [w for _, _, w in pairs]
    print(
        f"[debug] {wechat_id} samples={len(weights)}, half_life_days={half_life_days:.1f}, "
        f"min_w={min(weights):.4f}, max_w={max(weights):.4f}, "
        f"avg_w={sum(weights)/len(weights):.4f}"
    )

    weights_sorted = sorted(weights)

    def _percentile(values: List[float], ratio: float) -> float:
        if not values:
            return 0.0
        idx = max(0, min(len(values) - 1, int(ratio * (len(values) - 1))))
        return values[idx]

    p10 = _percentile(weights_sorted, 0.10)
    p50 = _percentile(weights_sorted, 0.50)
    p90 = _percentile(weights_sorted, 0.90)
    print(
        f"[debug] {wechat_id} w_p10={p10:.4f}, w_p50={p50:.4f}, w_p90={p90:.4f}"
    )

    all_texts = [ctx for ctx, _, _ in pairs] + [rep for _, rep, _ in pairs]
    vocab = Vocab.build(all_texts, max_size=2000, min_freq=1)

    dataset = StyleDataset(pairs, vocab=vocab, max_len=200)
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    def collate(batch):
        return collate_fn(batch, pad_id=vocab.pad_id)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = Seq2SeqGRU(vocab_size=len(vocab.itos)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir_path / f"{wechat_id}_vocab.json"
    model_path = out_dir_path / f"{wechat_id}_seq2seq.pt"

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src, tgt, sample_w in train_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            sample_w = sample_w.to(device)

            logits = model(src, tgt, teacher_forcing=0.7)
            target = tgt[:, 1:]

            B, Tm1, V = logits.shape
            loss_tokens = criterion(logits.reshape(-1, V), target.reshape(-1)).view(B, Tm1)

            mask = (target != vocab.pad_id).float()
            token_counts = mask.sum(dim=1).clamp_min(1.0)
            per_sample_loss = (loss_tokens * mask).sum(dim=1) / token_counts

            w_sum = sample_w.sum().clamp_min(1e-6)
            loss = (per_sample_loss * sample_w).sum() / w_sum

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * src.size(0)

        avg_train = total_loss / len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, _ in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)
                logits = model(src, tgt, teacher_forcing=0.0)

                target = tgt[:, 1:]
                B, Tm1, V = logits.shape
                loss_tokens = criterion(logits.reshape(-1, V), target.reshape(-1)).view(B, Tm1)

                mask = (target != vocab.pad_id).float()
                token_counts = mask.sum(dim=1).clamp_min(1.0)
                per_sample_loss = (loss_tokens * mask).sum(dim=1) / token_counts

                batch_loss = per_sample_loss.mean()
                val_loss += batch_loss.item() * src.size(0)
        avg_val = val_loss / len(val_ds)

        print(f"[epoch {epoch}] train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), model_path)
            vocab.save(vocab_path)
            print(f"  -> 新 best 模型已保存到 {model_path}")

    print("训练完成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a per-contact style GRU model.")
    parser.add_argument("--db", type=str, default="wechat_assistant.db", help="Path to SQLite DB")
    parser.add_argument("--wechat-id", type=str, required=True, help="Contact identifier for training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-dir", type=str, default="nn_models")
    parser.add_argument(
        "--half-life-days",
        type=float,
        default=120.0,
        help="Half-life in days for time-decay weighting (e.g., 60 for faster drift, 180 for slower)",
    )
    args = parser.parse_args()

    train_for_contact(
        db_path=args.db,
        wechat_id=args.wechat_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        default_half_life_days=args.half_life_days,
    )
