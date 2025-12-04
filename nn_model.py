from __future__ import annotations

"""GRU-based sequence-to-sequence model for style imitation."""

import torch
import torch.nn as nn


class Seq2SeqGRU(nn.Module):
    """Lightweight encoder-decoder built on GRUs."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, src_ids: torch.Tensor) -> torch.Tensor:
        """Encode a batch of source sequences into hidden state."""

        emb = self.emb(src_ids)
        _, h_n = self.encoder(emb)
        return h_n  # [num_layers, B, H]

    def decode_step(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a single decoding step given previous token IDs and hidden state."""

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        emb = self.emb(input_ids)
        output, hidden = self.decoder(emb, hidden)
        logits = self.out(output.squeeze(1))
        return logits, hidden

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing for training."""

        device = src_ids.device
        _ = device  # appease linters when device is unused
        batch_size, tgt_len = tgt_ids.shape

        hidden = self.encode(src_ids)

        inputs = tgt_ids[:, 0]
        logits_list = []
        for t in range(1, tgt_len):
            logits, hidden = self.decode_step(inputs, hidden)
            logits_list.append(logits.unsqueeze(1))

            use_teacher = torch.rand(1).item() < teacher_forcing
            if use_teacher:
                inputs = tgt_ids[:, t]
            else:
                inputs = torch.argmax(logits, dim=-1)

        return torch.cat(logits_list, dim=1)
