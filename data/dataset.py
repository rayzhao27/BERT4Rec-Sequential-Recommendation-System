"""
data/dataset.py
───────────────
PyTorch Dataset and DataLoader factory for BERT4Rec.

Three dataset modes mirror the leave-one-out split:

    TRAIN  mode  ─  random BERT-style masking over seq[:-2]
    VAL    mode  ─  mask only the last item of seq[:-1] (fixed evaluation)
    TEST   mode  ─  mask only the last item of seq      (fixed evaluation)

Masking strategy (identical to the original BERT4Rec paper, Sun et al. 2019):
    • With probability mask_prob, replace an item with [MASK].
    • With probability 0.1 of the masked set, replace with a random item.
    • With probability 0.1 of the masked set, keep the original item.
    • The remaining 80 % are replaced with [MASK].
    (Only applied during training; evaluation always masks the final position.)

Special tokens:
    PAD   = 0                    padding token
    MASK  = num_items + 1        [MASK] token  (always the last vocab slot)

Output batch tensors (all shape [B, max_seq_len]):
    input_ids      padded + masked item sequence
    labels         original item ids at masked positions, -100 elsewhere
    padding_mask   1 where token is PAD (for attention key_padding_mask)
"""

from __future__ import annotations

import random
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import joblib

from torch import Tensor
from torch.utils.data import DataLoader, Dataset


# ── Token constants ────────────────────────────────────────────────────────────
PAD_ID   = 0
IGNORE_ID = -100   # label value ignored by CrossEntropyLoss


class Split(str, Enum):
    TRAIN = "train"
    VAL   = "val"
    TEST  = "test"


# ── Dataset ───────────────────────────────────────────────────────────────────

class BERT4RecDataset(Dataset):
    """
    Args:
        sequences:    dict mapping user_id → list[item_id] (1-based).
        num_items:    total number of distinct items (PAD=0, MASK=num_items+1).
        max_seq_len:  maximum sequence length (sequences are right-truncated
                      then left-padded to this length).
        mask_prob:    fraction of positions randomly masked (training only).
        split:        Split.TRAIN | Split.VAL | Split.TEST
        seed:         random seed for reproducible masking (training).
    """

    def __init__(
        self,
        sequences:   dict[int, list[int]],
        num_items:   int,
        max_seq_len: int  = 200,
        mask_prob:   float = 0.2,
        split:       Split = Split.TRAIN,
        seed:        int   = 42,
        item_features: dict[int, dict] | None = None,
        num_genres: int = 0,
        num_decades: int = 0,
    ) -> None:
        self.sequences   = list(sequences.values())
        self.user_ids    = list(sequences.keys())
        self.num_items   = num_items
        self.max_seq_len = max_seq_len
        self.mask_prob   = mask_prob
        self.split       = split
        self.mask_id     = num_items + 1   # [MASK] token id
        self._rng        = random.Random(seed)

        # ── side features ──
        self.item_features = item_features
        self.num_genres = num_genres
        self.num_decades = num_decades

        # Pre-build lookup tables for fast indexing.
        # Row 0 is reserved for PAD / MASK (all zeros).
        # Rows 1..num_items+1 hold real item features; row mask_id stays zero too.
        if item_features is not None:
            table_size = num_items + 2  # PAD + items + MASK
            self._genre_table = torch.zeros(table_size, num_genres, dtype=torch.float)
            self._decade_table = torch.zeros(table_size, dtype=torch.long)
            for item_id, feats in item_features.items():
                self._genre_table[item_id] = torch.tensor(feats["genres"], dtype=torch.float)
                self._decade_table[item_id] = feats["decade"]

    # ── helpers ────────────────────────────────────────────────────────────────

    def _truncate_and_pad(self, seq: list[int]) -> list[int]:
        """Right-truncate to max_seq_len, then left-pad with PAD_ID."""
        seq = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(seq)
        return [PAD_ID] * pad_len + seq

    def _mask_train(self, seq: list[int]) -> tuple[list[int], list[int]]:
        """
        Apply BERT-style random masking to *seq* (training mode).
        Returns (masked_seq, labels) where labels[i] = original item if masked,
        else IGNORE_ID.
        """
        masked = seq.copy()
        labels = [IGNORE_ID] * len(seq)

        for i, item in enumerate(seq):
            if item == PAD_ID:
                continue
            if self._rng.random() < self.mask_prob:
                labels[i] = item
                r = self._rng.random()
                if r < 0.8:
                    masked[i] = self.mask_id                              # 80 % → [MASK]
                elif r < 0.9:
                    masked[i] = self._rng.randint(1, self.num_items)     # 10 % → random item
                # else: 10 % → keep original (masked[i] unchanged)

        return masked, labels

    def _mask_eval(self, seq: list[int]) -> tuple[list[int], list[int]]:
        """
        Mask only the final non-PAD position (val / test mode).
        The model must predict the target item at that single position.
        """
        masked = seq.copy()
        labels = [IGNORE_ID] * len(seq)

        # Find last real (non-PAD) position
        for i in range(len(seq) - 1, -1, -1):
            if seq[i] != PAD_ID:
                labels[i] = seq[i]
                masked[i] = self.mask_id
                break

        return masked, labels

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raw_seq = self.sequences[idx]
        seq     = self._truncate_and_pad(raw_seq)

        if self.split == Split.TRAIN:
            input_ids, labels = self._mask_train(seq)
        else:
            input_ids, labels = self._mask_eval(seq)

        input_ids_t  = torch.tensor(input_ids, dtype=torch.long)
        labels_t     = torch.tensor(labels,    dtype=torch.long)
        padding_mask = (input_ids_t == PAD_ID)

        out = {
            "input_ids":    input_ids_t,
            "labels":       labels_t,
            "padding_mask": padding_mask,
            "user_id":      torch.tensor(self.user_ids[idx], dtype=torch.long),
        }

        # ── side features (look up by input_ids) ──
        if self.item_features is not None:
            out["genres"] = self._genre_table[input_ids_t]    # [L, num_genres]
            out["decade"] = self._decade_table[input_ids_t]   # [L]

        return out


# ── DataLoader factory ─────────────────────────────────────────────────────────

def build_dataloaders(
    processed_dir: str | Path,
    max_seq_len:   int   = 200,
    mask_prob:     float = 0.2,
    batch_size:    int   = 256,
    num_workers:   int   = 4,
    seed:          int   = 42,
    use_features:  bool  = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, Any]]:
    processed_dir = Path(processed_dir)

    train_seqs = joblib.load(processed_dir / "train_seqs.pkl")
    val_seqs   = joblib.load(processed_dir / "val_seqs.pkl")
    test_seqs  = joblib.load(processed_dir / "test_seqs.pkl")
    item_enc   = joblib.load(processed_dir / "item_encoder.pkl")

    num_items = len(item_enc.classes_)

    # ── Side features (optional) ──
    item_features = None
    num_genres    = 0
    num_decades   = 0
    if use_features:
        feat_path = processed_dir / "item_features.pkl"
        if feat_path.exists():
            item_features = joblib.load(feat_path)
            import json
            stats_json = json.loads((processed_dir / "stats.json").read_text())
            num_genres  = stats_json["num_genres"]
            num_decades = stats_json["num_decades"]

    common = dict(
        num_items     = num_items,
        max_seq_len   = max_seq_len,
        seed          = seed,
        item_features = item_features,
        num_genres    = num_genres,
        num_decades   = num_decades,
    )

    train_ds = BERT4RecDataset(train_seqs, mask_prob=mask_prob, split=Split.TRAIN, **common)
    val_ds   = BERT4RecDataset(val_seqs,   mask_prob=mask_prob, split=Split.VAL,   **common)
    test_ds  = BERT4RecDataset(test_seqs,  mask_prob=mask_prob, split=Split.TEST,  **common)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    stats = {
        "num_items":       num_items,
        "vocab_size":      num_items + 2,
        "mask_token_id":   num_items + 1,
        "num_train_users": len(train_seqs),
        "num_val_users":   len(val_seqs),
        "num_test_users":  len(test_seqs),
        "train_batches":   len(train_loader),
        "num_genres":      num_genres,
        "num_decades":     num_decades,
        "use_features":    item_features is not None,
    }
    return train_loader, val_loader, test_loader, stats


# ── Quick smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import logging

    logging.basicConfig(level="INFO", format="%(levelname)s | %(message)s")

    train_loader, val_loader, test_loader, stats = build_dataloaders(
        processed_dir="data/processed",
        max_seq_len=200,
        mask_prob=0.2,
        batch_size=256,
    )

    print("\n── Dataset stats ─────────────────────────────────────")
    print(json.dumps(stats, indent=2))

    # Inspect a single batch
    batch = next(iter(train_loader))
    print("\n── Train batch shapes ────────────────────────────────")
    for k, v in batch.items():
        print(f"  {k:15s}  {tuple(v.shape)}  dtype={v.dtype}")

    # Confirm [MASK] tokens appear in input_ids
    mask_id      = stats["mask_token_id"]
    n_masked     = (batch["input_ids"] == mask_id).sum().item()
    n_labels     = (batch["labels"]    != IGNORE_ID).sum().item()
    n_pad        = batch["padding_mask"].sum().item()
    print(f"\n  [MASK] tokens in batch : {n_masked}")
    print(f"  labelled positions     : {n_labels}")
    print(f"  [PAD]  tokens in batch : {n_pad}")

    assert n_masked <= n_labels, "More [MASK] tokens than labelled positions!"
    mask_ratio = n_masked / n_labels
    assert 0.70 <= mask_ratio <= 0.90, f"Unexpected mask ratio: {mask_ratio:.2f} (expected ~0.80)"

    print(f"\n  [MASK] covers {mask_ratio:.1%} of labelled positions (expect ~80%)")
    print("\n  Smoke-test passed ✓")

    print(f"\n  genres shape: {batch['genres'].shape}")  # 应该是 [256, 200, 18]
    print(f"  decade shape: {batch['decade'].shape}")  # 应该是 [256, 200]
    print(f"  genres at PAD pos: {batch['genres'][0, 0]}")  # 全 0
    print(f"  genres at real pos: {batch['genres'][0, -1]}")  # 非全 0（除非最后是 MASK）
