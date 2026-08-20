from __future__ import annotations

"""
model/bert4rec.py
─────────────────
Full BERT4Rec model (Sun et al., 2019).

Architecture:
    input_ids  [B, L]
        │
    BERTEmbeddings          item emb + positional emb → LayerNorm → Dropout
        │
    TransformerBlock × N    bidirectional self-attention + FFN  (N=2 default)
        │
    PredictionHead          LayerNorm → Linear(d → vocab_size)
        │
    logits  [B, L, vocab_size]     (loss computed only at [MASK] positions)

At inference, only the last position is masked and the top-K logit indices
are returned as the recommended item ids.
"""

import torch
import torch.nn as nn
from torch import Tensor

from model.attention import TransformerBlock
from model.embeddings import BERTEmbeddings


class PredictionHead(nn.Module):
    """
    Thin output head: LayerNorm → GELU → Linear(d → vocab_size).

    Mirrors the masked language model head in HuggingFace BERT and gives
    a small but consistent improvement over a bare linear projection.
    """

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense     = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.decoder   = nn.Linear(hidden_size, vocab_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.gelu(self.dense(x))
        x = self.layer_norm(x)
        return self.decoder(x)   # [B, L, vocab_size]


class BERT4Rec(nn.Module):
    """
    BERT4Rec sequential recommendation model.

    Args:
        vocab_size:                  num_items + 2  (PAD=0, MASK=num_items+1).
        hidden_size:                 model/embedding dimension d  (default 256).
        max_seq_len:                 maximum interaction sequence length (default 200).
        num_hidden_layers:           number of stacked transformer blocks (default 2).
        num_attention_heads:         attention heads per block  (default 4).
        intermediate_size:           FFN inner dimension  (default 1024 = 4×d).
        hidden_dropout_prob:         dropout on embeddings and FFN outputs (default 0.1).
        attention_probs_dropout:     dropout on attention weights (default 0.1).
        pad_token_id:                id of the [PAD] token (default 0).
    """

    def __init__(
        self,
        vocab_size:                int,
        hidden_size:               int   = 256,
        max_seq_len:               int   = 200,
        num_hidden_layers:         int   = 2,
        num_attention_heads:       int   = 4,
        intermediate_size:         int   = 1024,
        hidden_dropout_prob:       float = 0.1,
        attention_probs_dropout:   float = 0.1,
        pad_token_id:              int   = 0,
        num_genres:                int = 0,
        num_decades:               int = 0,
    ) -> None:
        super().__init__()

        self.pad_token_id = pad_token_id
        self.hidden_size  = hidden_size
        self.vocab_size   = vocab_size

        # ── 1. Embedding layer ───────────────────────────────────────────────
        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_seq_len=max_seq_len,
            dropout=hidden_dropout_prob,
            pad_token_id=pad_token_id,
            num_genres=num_genres,
            num_decades=num_decades,
        )

        # ── 2. Transformer stack ─────────────────────────────────────────────
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size             = hidden_size,
                num_attention_heads     = num_attention_heads,
                intermediate_size       = intermediate_size,
                hidden_dropout_prob     = hidden_dropout_prob,
                attention_probs_dropout = attention_probs_dropout,
            )
            for _ in range(num_hidden_layers)
        ])

        # ── 3. Output head ───────────────────────────────────────────────────
        self.head = PredictionHead(hidden_size, vocab_size)

        # Weight initialisation (follows original BERT)
        self.apply(self._init_weights)

    # ── Weight initialisation ────────────────────────────────────────────────

    def _init_weights(self, module: nn.Module) -> None:
        """Truncated normal for linear/embedding weights; zero bias."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:    Tensor,
        padding_mask: Tensor | None = None,
        genres:       Tensor | None = None,
        decade:       Tensor | None = None,
    ) -> Tensor:
        """
        Full forward pass used during training.

        Args:
            input_ids:    LongTensor  [B, L]  — masked item-id sequence.
            padding_mask: BoolTensor  [B, L]  — True where token is PAD.
                          Derived from input_ids automatically if not supplied.

        Returns:
            logits: FloatTensor [B, L, vocab_size]
            Loss is computed externally (see training/loss.py).
        """
        # Derive padding mask from input_ids when not explicitly provided
        if padding_mask is None:
            padding_mask = (input_ids == self.pad_token_id)   # [B, L]

        # ── Embeddings ───────────────────────────────────────────────────────
        x = self.embeddings(input_ids, genres=genres, decade=decade)         # [B, L, d]

        # ── Transformer layers ───────────────────────────────────────────────
        for block in self.transformer_blocks:
            x = block(x, padding_mask=padding_mask)   # [B, L, d]

        # ── Prediction head ──────────────────────────────────────────────────
        logits = self.head(x)                  # [B, L, vocab_size]
        return logits

    # ── Inference helper ─────────────────────────────────────────────────────

    @torch.inference_mode()
    def recommend(
        self,
        input_ids:      Tensor,
        mask_token_id:  int,
        top_k:          int = 10,
        filter_seen:    bool = True,
    ) -> tuple[Tensor, Tensor]:
        """
        Recommend the top-K items for each sequence in the batch.

        Masks the last real (non-PAD) position, runs a forward pass,
        and returns the top-K item ids and their raw logit scores.

        Args:
            input_ids:     LongTensor [B, L]  — user history (0 = PAD).
            mask_token_id: id of the [MASK] token.
            top_k:         number of items to return per user.
            filter_seen:   if True, zero-out logits for items already in
                           the input sequence (no re-recommendations).

        Returns:
            top_ids:    LongTensor  [B, top_k]  — recommended item ids.
            top_scores: FloatTensor [B, top_k]  — corresponding logit scores.
        """
        self.eval()
        B, L = input_ids.shape
        device = input_ids.device

        # ── Find the last non-PAD position for each sequence ─────────────────
        # last_pos[b] = index of the final real token in sequence b
        non_pad     = (input_ids != self.pad_token_id).long()   # [B, L]
        last_pos    = (non_pad * torch.arange(L, device=device)).argmax(dim=1)  # [B]

        # ── Insert [MASK] at last_pos ─────────────────────────────────────────
        masked_input = input_ids.clone()
        masked_input[torch.arange(B, device=device), last_pos] = mask_token_id

        # ── Forward pass ──────────────────────────────────────────────────────
        padding_mask = (masked_input == self.pad_token_id)
        logits       = self.forward(masked_input, padding_mask)   # [B, L, V]

        # ── Extract logits at the masked position ─────────────────────────────
        # gather shape: [B, 1, V] → squeeze → [B, V]
        idx            = last_pos.view(B, 1, 1).expand(B, 1, self.vocab_size)
        masked_logits  = logits.gather(dim=1, index=idx).squeeze(1)   # [B, V]

        # ── Filter already-seen items ─────────────────────────────────────────
        if filter_seen:
            for b in range(B):
                seen = input_ids[b].unique()
                seen = seen[seen != self.pad_token_id]
                masked_logits[b, seen] = float("-inf")

        # ── Top-K ────────────────────────────────────────────────────────────
        top_scores, top_ids = torch.topk(masked_logits, k=top_k, dim=-1)
        return top_ids, top_scores

    # ── Parameter count helper ───────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Return total (or trainable-only) parameter count."""
        params = (
            self.parameters()
            if not trainable_only
            else filter(lambda p: p.requires_grad, self.parameters())
        )
        return sum(p.numel() for p in params)


# ── Factory from config dict / Hydra OmegaConf ───────────────────────────────

def build_model(cfg) -> BERT4Rec:
    """
    Instantiate BERT4Rec from a config object (dict, Namespace, or OmegaConf).

    Supports both dict-style (cfg['hidden_size']) and attr-style (cfg.hidden_size)
    access, so it works with plain dicts, argparse Namespace, and Hydra configs.
    """
    def get(key, default=None):
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    return BERT4Rec(
        vocab_size               = get("vocab_size"),
        hidden_size              = get("hidden_size",             256),
        max_seq_len              = get("max_seq_len",             200),
        num_hidden_layers        = get("num_hidden_layers",       2),
        num_attention_heads      = get("num_attention_heads",     4),
        intermediate_size        = get("intermediate_size",       1024),
        hidden_dropout_prob      = get("hidden_dropout_prob",     0.1),
        attention_probs_dropout  = get("attention_probs_dropout", 0.1),
        pad_token_id             = get("pad_token_id",            0),
        num_genres               =get("num_genres",               0),
        num_decades              =get("num_decades",              0),
    )


# ── Smoke-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import torch

    print("Running BERT4Rec smoke-test …\n")

    cfg = {
        "vocab_size":              3535,   # 3533 items + PAD + MASK
        "hidden_size":             256,
        "max_seq_len":             200,
        "num_hidden_layers":       2,
        "num_attention_heads":     4,
        "intermediate_size":       1024,
        "hidden_dropout_prob":     0.1,
        "attention_probs_dropout": 0.1,
    }

    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Device: {device}\n")

    model = build_model(cfg).to(device)
    print(f"Parameters: {model.num_parameters():,}\n")

    # ── Training forward pass ────────────────────────────────────────────────
    B, L   = 4, 200
    MASK   = cfg["vocab_size"] - 1   # 3534
    PAD    = 0

    input_ids    = torch.randint(1, cfg["vocab_size"] - 1, (B, L), device=device)
    padding_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

    # Inject some padding and masks
    input_ids[:, :20]    = PAD
    padding_mask[:, :20] = True
    input_ids[:, 50]     = MASK
    input_ids[:, 120]    = MASK

    logits = model(input_ids, padding_mask)
    print(f"Logits shape:  {tuple(logits.shape)}  (expect [{B}, {L}, {cfg['vocab_size']}])")

    # ── Inference ────────────────────────────────────────────────────────────
    top_ids, top_scores = model.recommend(input_ids, mask_token_id=MASK, top_k=10)
    print(f"Top-10 recs shape: {tuple(top_ids.shape)}  (expect [{B}, 10])")
    print(f"Sample recs (user 0): {top_ids[0].tolist()}\n")

    print("Smoke-test passed ✓")
