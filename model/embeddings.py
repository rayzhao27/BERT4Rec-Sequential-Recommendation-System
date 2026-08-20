from __future__ import annotations

"""
model/embeddings.py
───────────────────
Embedding layer for BERT4Rec with optional side features.

If side features are enabled, the per-item representation is built as:
    item_emb    = ItemEmbedding(item_id)              [d_item]
    genre_emb   = Linear(num_genres → d_genre)(genres) [d_genre]
    decade_emb  = DecadeEmbedding(decade)              [d_decade]
    fused       = Linear([item_emb, genre_emb, decade_emb] → hidden_size)
Otherwise, fused = ItemEmbedding(item_id) (hidden_size).

Final output: LayerNorm(fused + positional_emb) → Dropout
"""

import torch
import torch.nn as nn
from torch import Tensor


class BERTEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        hidden_size:   int,
        max_seq_len:   int,
        dropout:       float = 0.1,
        pad_token_id:  int   = 0,
        # ── side features ──
        num_genres:    int   = 0,
        num_decades:   int   = 0,
        d_item:        int   = 192,
        d_genre:       int   = 32,
        d_decade:      int   = 32,
    ) -> None:
        super().__init__()
        self.pad_token_id  = pad_token_id
        self.use_features  = num_genres > 0 and num_decades > 0

        if self.use_features:
            # ── feature-aware item representation ──
            self.item_embeddings   = nn.Embedding(vocab_size, d_item, padding_idx=pad_token_id)
            self.genre_projection  = nn.Linear(num_genres, d_genre)
            self.decade_embeddings = nn.Embedding(num_decades + 1, d_decade, padding_idx=0)
            #                                     ^^^ +1 because we reserve idx 0 for PAD
            # NOTE: decade ids in dataset are 0-based; we shift them by +1 here.

            fused_in = d_item + d_genre + d_decade
            self.feature_fuse = nn.Linear(fused_in, hidden_size)
        else:
            # ── plain item embedding ──
            self.item_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)

        self.position_embeddings = nn.Embedding(max_seq_len + 1, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: Tensor,
        genres:    Tensor | None = None,   # [B, L, num_genres] float
        decade:    Tensor | None = None,   # [B, L]            long
    ) -> Tensor:
        B, L   = input_ids.shape
        device = input_ids.device
        pad_mask = (input_ids == self.pad_token_id)   # [B, L]

        # ── item representation ──
        if self.use_features:
            assert genres is not None and decade is not None, \
                "Side features required when use_features=True"

            item_emb   = self.item_embeddings(input_ids)              # [B, L, d_item]
            genre_emb  = self.genre_projection(genres)                # [B, L, d_genre]

            # shift decade by +1 so that 0 stays as PAD slot
            decade_shifted = decade + 1
            decade_shifted = decade_shifted.masked_fill(pad_mask, 0)  # PAD → 0
            decade_emb = self.decade_embeddings(decade_shifted)       # [B, L, d_decade]

            fused = torch.cat([item_emb, genre_emb, decade_emb], dim=-1)
            fused = self.feature_fuse(fused)                          # [B, L, hidden]

            # Zero out PAD rows just to be safe (item_emb has padding_idx but
            # genre/decade projections don't).
            fused = fused.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        else:
            fused = self.item_embeddings(input_ids)                   # [B, L, hidden]

        # ── positional embedding ──
        positions = torch.arange(1, L + 1, device=device).unsqueeze(0)   # [1, L]
        positions = positions.expand(B, -1).masked_fill(pad_mask, 0)
        pos_emb   = self.position_embeddings(positions)                  # [B, L, hidden]

        # ── combine ──
        out = fused + pos_emb
        out = self.layer_norm(out)
        out = self.dropout(out)
        return out