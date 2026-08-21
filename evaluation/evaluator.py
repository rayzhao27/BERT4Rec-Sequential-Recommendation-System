from __future__ import annotations

"""
evaluation/evaluator.py
────────────────────────
Full offline evaluation pipeline for BERT4Rec.

Protocol (identical to the original paper):
  1. Load the test split — each user's full interaction history.
  2. Mask the last item (the held-out target).
  3. Run the model to get logits over all vocab items.
  4. Zero-out logits for items the user already interacted with
     (no re-recommending seen items).
  5. Rank remaining items by logit score descending.
  6. Check whether the target falls within the top K.
  7. Average HR@K and NDCG@K across all users.

Usage:
    python evaluation/evaluator.py --data_dir data \
                                   --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import BERT4RecDataset, Split
from model.bert4rec import BERT4Rec, build_model
from evaluation.metrics import MetricAccumulator

logger = logging.getLogger(__name__)


# ── Checkpoint loader ─────────────────────────────────────────────────────────

def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device:          torch.device,
) -> tuple[BERT4Rec, dict]:
    """
    Load model weights and config from a trainer checkpoint.

    Returns:
        model:  BERT4Rec in eval mode on *device*.
        cfg:    the config dict stored in the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["cfg"]

    # cfg from trainer doesn't include vocab_size — load from stats
    data_dir  = Path(cfg["data_dir"])
    stats = json.loads((data_dir / "processed" / "stats.json").read_text())
    cfg["vocab_size"] = stats["num_items"] + 2  # PAD + items + MASK

    # If the checkpoint was trained with side features, restore those dims too
    if cfg.get("use_features", False):
        cfg["num_genres"] = stats["num_genres"]
        cfg["num_decades"] = stats["num_decades"]

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    logger.info(
        "Loaded checkpoint from %s  (epoch %d, val_loss %.4f)",
        checkpoint_path, ckpt["epoch"], ckpt["best_loss"],
    )
    return model, cfg


# ── Core evaluation loop ──────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:         BERT4Rec,
    test_loader:   DataLoader,
    mask_token_id: int,
    k_values:      list[int] = [5, 10, 20],
    device:        torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """
    Run full evaluation over *test_loader*.

    For each user:
      • The DataLoader already provides input_ids with the last real item
        replaced by [MASK] and labels[last_pos] = target item.
      • We get logits from the model, zero-out seen items, rank the rest,
        then compute HR@K and NDCG@K.

    Returns:
        dict of metric name → value, e.g. {'HR@10': 0.312, 'NDCG@10': 0.198}
    """
    model.eval()
    acc = MetricAccumulator(k_values=k_values)

    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        genres = batch["genres"].to(device) if "genres" in batch else None
        decade = batch["decade"].to(device) if "decade" in batch else None

        logits = model(input_ids, padding_mask, genres=genres, decade=decade)

        B, L, V = logits.shape

        target_pos = (labels != -100).long().argmax(dim=1)
        target_ids = labels[torch.arange(B, device=device), target_pos]

        idx = target_pos.view(B, 1, 1).expand(B, 1, V)
        target_logits = logits.gather(1, idx).squeeze(1)

        for b in range(B):
            seen = input_ids[b]
            seen = seen[(seen != 0) & (seen != mask_token_id)]
            target_logits[b, seen] = float("-inf")

        ranked = torch.argsort(target_logits, dim=-1, descending=True)

        ranked_lists = ranked.cpu().tolist()
        target_list = target_ids.cpu().tolist()
        acc.update(ranked_lists, target_list)

    return acc.compute()


# ── Entry point ───────────────────────────────────────────────────────────────

def run_evaluation(
    checkpoint_path: str | Path,
    data_dir:        str | Path,
    k_values:        list[int] = [5, 10, 20],
    batch_size:      int       = 256,
    num_workers:     int       = 0,
) -> dict[str, float]:
    """
    Load a checkpoint, build the test DataLoader, and return metrics.
    """
    device = (
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    logger.info("Device: %s", device)

    # ── Model ─────────────────────────────────────────────────────────────
    model, cfg = load_model_from_checkpoint(checkpoint_path, device)

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir = Path(data_dir)
    proc_dir = data_dir / "processed"

    test_seqs = joblib.load(proc_dir / "test_seqs.pkl")
    item_enc = joblib.load(proc_dir / "item_encoder.pkl")
    stats = json.loads((proc_dir / "stats.json").read_text())

    num_items = stats["num_items"]
    mask_token_id = stats["vocab_size"] - 1

    # ── Load side features if the model was trained with them ──
    use_features = cfg.get("use_features", False) or bool(cfg.get("num_genres", 0))
    item_features = None
    num_genres = num_decades = 0
    if use_features:
        feat_path = proc_dir / "item_features.pkl"
        if feat_path.exists():
            item_features = joblib.load(feat_path)
            num_genres = stats["num_genres"]
            num_decades = stats["num_decades"]
            logger.info("Loaded side features for evaluation (genres=%d, decades=%d)",
                        num_genres, num_decades)

    test_ds = BERT4RecDataset(
        sequences=test_seqs,
        num_items=num_items,
        max_seq_len=cfg.get("max_seq_len", 200),
        mask_prob=cfg.get("mask_prob", 0.2),
        split=Split.TEST,
        item_features=item_features,
        num_genres=num_genres,
        num_decades=num_decades,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = False,
    )

    logger.info("Test users: %d  |  batches: %d", len(test_ds), len(test_loader))

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics = evaluate(
        model         = model,
        test_loader   = test_loader,
        mask_token_id = mask_token_id,
        k_values      = k_values,
        device        = device,
    )

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate BERT4Rec — HR@K and NDCG@K")
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--data_dir",   default="data")
    p.add_argument("--k_values",   nargs="+", type=int, default=[5, 10, 20])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers",type=int, default=0)
    p.add_argument("--log_level",  default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level   = args.log_level,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt = "%H:%M:%S",
    )

    metrics = run_evaluation(
        checkpoint_path = args.checkpoint,
        data_dir        = args.data_dir,
        k_values        = args.k_values,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
    )

    print("\n── Evaluation Results ─────────────────────────────────────")
    for k in args.k_values:
        print(f"  HR@{k:<3}   {metrics[f'HR@{k}']:.4f}")
        print(f"  NDCG@{k:<3} {metrics[f'NDCG@{k}']:.4f}")
        print(f"  MRR@{k:<3}  {metrics[f'MRR@{k}']:.4f}")
        print()

    # Save results to JSON
    out_path = Path("evaluation_results.json")
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"Results saved → {out_path}")
