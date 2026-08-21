from __future__ import annotations

"""
training/trainer.py
────────────────────
Training and validation loop for BERT4Rec.

Features:
  • Mixed-precision (AMP) on CUDA; falls back cleanly on MPS / CPU
  • Gradient clipping to prevent exploding gradients
  • TensorBoard logging (loss, ppl, lr, grad-norm per step)
  • Best-model checkpointing by validation loss
  • Resume from checkpoint
  • Clean keyboard-interrupt handling (saves checkpoint before exit)

Usage:
    python training/trainer.py --data_dir data --epochs 200
"""

import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import build_dataloaders
from model.bert4rec import BERT4Rec, build_model
from training.loss import MaskedItemLoss, masked_accuracy
from training.scheduler import get_scheduler, compute_total_steps

logger = logging.getLogger(__name__)


# ── Device selection ──────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch:     int,
    step:      int,
    best_loss: float,
    cfg:       dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "step":       step,
        "best_loss":  best_loss,
        "cfg":        cfg,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict(),
    }, path)
    logger.info("  checkpoint saved → %s", path)


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device:    torch.device,
) -> tuple[int, int, float]:
    """Returns (start_epoch, global_step, best_loss)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    logger.info("Resumed from %s  (epoch %d, step %d)", path, ckpt["epoch"], ckpt["step"])
    return ckpt["epoch"] + 1, ckpt["step"], ckpt["best_loss"]


# ── One training epoch ────────────────────────────────────────────────────────

def train_epoch(
    model:      BERT4Rec,
    loader:     DataLoader,
    criterion:  MaskedItemLoss,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    scaler:     GradScaler | None,
    device:     torch.device,
    writer:     SummaryWriter,
    global_step: int,
    grad_clip:  float,
    use_amp:    bool,
    epoch:      int,
) -> tuple[float, int]:
    """
    Run one full pass over the training DataLoader.

    Returns:
        mean_loss:   average loss over the epoch.
        global_step: updated step counter.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0         = time.time()

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        # side features (optional)
        genres = batch["genres"].to(device) if "genres" in batch else None
        decade = batch["decade"].to(device) if "decade" in batch else None

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(input_ids, padding_mask, genres=genres, decade=decade)
                loss, metrics = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, padding_mask, genres=genres, decade=decade)
            loss, metrics = criterion(logits, labels)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        scheduler.step()
        global_step += 1

        total_loss += metrics["loss"]
        n_batches += 1

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("train/loss", metrics["loss"], global_step)
        writer.add_scalar("train/ppl", metrics["ppl"], global_step)
        writer.add_scalar("train/lr", lr, global_step)
        writer.add_scalar("train/grad_norm", grad_norm, global_step)

        if global_step % 50 == 0:
            acc = masked_accuracy(logits.detach(), labels)
            writer.add_scalar("train/masked_acc", acc, global_step)
            elapsed = time.time() - t0
            logger.info(
                "  epoch %d | step %6d | loss %.4f | ppl %6.1f | "
                "acc %.3f | lr %.2e | %.1fs",
                epoch, global_step, metrics["loss"], metrics["ppl"],
                acc, lr, elapsed,
            )

    return total_loss / max(n_batches, 1), global_step


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     BERT4Rec,
    loader:    DataLoader,
    criterion: MaskedItemLoss,
    device:    torch.device,
    writer:    SummaryWriter,
    step:      int,
    epoch:     int,
) -> float:
    """
    Evaluate on the validation set.  Returns mean validation loss.
    Validation always masks the last real item per sequence (deterministic).
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        genres = batch["genres"].to(device) if "genres" in batch else None
        decade = batch["decade"].to(device) if "decade" in batch else None

        logits = model(input_ids, padding_mask, genres=genres, decade=decade)
        _, metrics = criterion(logits, labels)
        acc = masked_accuracy(logits, labels)

        total_loss += metrics["loss"]
        total_acc += acc
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    mean_acc  = total_acc  / max(n_batches, 1)
    mean_ppl  = torch.exp(torch.tensor(mean_loss)).item()

    writer.add_scalar("val/loss", mean_loss, step)
    writer.add_scalar("val/ppl",  mean_ppl,  step)
    writer.add_scalar("val/masked_acc", mean_acc, step)

    logger.info(
        "── val epoch %d | loss %.4f | ppl %6.1f | acc %.3f",
        epoch, mean_loss, mean_ppl, mean_acc,
    )
    return mean_loss


# ── Main trainer ──────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    """
    Full training loop.

    Expected keys in *cfg*:
        data_dir, epochs, batch_size, learning_rate, weight_decay,
        warmup_steps, grad_clip, max_seq_len, mask_prob,
        num_workers, checkpoint_dir, resume, log_dir,
        hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_dropout_prob, attention_probs_dropout,
        label_smoothing
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    train_start = time.time()
    device = get_device()
    use_amp = (device.type == "cuda")          # AMP only on CUDA
    logger.info("Device: %s  |  AMP: %s", device, use_amp)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt   = ckpt_dir / "best_model.pt"
    latest_ckpt = ckpt_dir / "latest.pt"

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, stats = build_dataloaders(
        processed_dir = Path(cfg["data_dir"]) / "processed",
        max_seq_len   = cfg["max_seq_len"],
        mask_prob     = cfg["mask_prob"],
        batch_size    = cfg["batch_size"],
        num_workers   = cfg["num_workers"],
        seed          = cfg.get("seed", 42),
        use_features=cfg.get("use_features", False),
    )
    logger.info("Dataset: %s", stats)

    # ── Model ─────────────────────────────────────────────────────────────────
    model_cfg = {
        **cfg,
        "vocab_size": stats["vocab_size"],
        "num_genres": stats.get("num_genres", 0),
        "num_decades": stats.get("num_decades", 0),
    }
    model = build_model(model_cfg).to(device)

    # Write feature dims back into cfg so checkpoints record them.
    cfg["num_genres"] = model_cfg["num_genres"]
    cfg["num_decades"] = model_cfg["num_decades"]
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = MaskedItemLoss(
        vocab_size      = stats["vocab_size"],
        label_smoothing = cfg.get("label_smoothing", 0.0),
    )

    # AdamW with weight decay applied only to non-bias / non-LayerNorm params
    decay_params     = []
    no_decay_params  = []
    for name, param in model.named_parameters():
        if any(nd in name for nd in ("bias", "layer_norm", "LayerNorm")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg["learning_rate"], betas=(0.9, 0.999))

    total_steps = compute_total_steps(cfg["epochs"], len(train_loader))
    scheduler   = get_scheduler(
        optimizer    = optimizer,
        warmup_steps = cfg["warmup_steps"],
        total_steps  = total_steps,
        lr_min_ratio = cfg.get("lr_min_ratio", 0.1),
    )

    scaler = GradScaler() if use_amp else None

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_loss   = float("inf")

    resume_path = cfg.get("resume")
    if resume_path and Path(resume_path).exists():
        start_epoch, global_step, best_loss = load_checkpoint(
            Path(resume_path), model, optimizer, scheduler, device
        )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=cfg.get("log_dir", "runs/bert4rec"))

    # ── Graceful interrupt: save latest on Ctrl-C ─────────────────────────────
    def _handle_interrupt(sig, frame):
        logger.warning("Interrupted — saving latest checkpoint …")
        save_checkpoint(latest_ckpt, model, optimizer, scheduler,
                        start_epoch, global_step, best_loss, cfg)
        writer.close()
        sys.exit(0)
    signal.signal(signal.SIGINT, _handle_interrupt)

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("Starting training: epochs=%d, steps/epoch=%d, total=%d",
                cfg["epochs"], len(train_loader), total_steps)

    for epoch in range(start_epoch, cfg["epochs"]):
        t_epoch = time.time()

        # train
        train_loss, global_step = train_epoch(
            model       = model,
            loader      = train_loader,
            criterion   = criterion,
            optimizer   = optimizer,
            scheduler   = scheduler,
            scaler      = scaler,
            device      = device,
            writer      = writer,
            global_step = global_step,
            grad_clip   = cfg.get("grad_clip", 5.0),
            use_amp     = use_amp,
            epoch       = epoch,
        )

        # validate
        val_loss = validate(
            model     = model,
            loader    = val_loader,
            criterion = criterion,
            device    = device,
            writer    = writer,
            step      = global_step,
            epoch     = epoch,
        )

        epoch_time = time.time() - t_epoch
        logger.info(
            "Epoch %d/%d done in %.1fs | train_loss=%.4f | val_loss=%.4f",
            epoch + 1, cfg["epochs"], epoch_time, train_loss, val_loss,
        )

        # always save latest
        save_checkpoint(latest_ckpt, model, optimizer, scheduler,
                        epoch, global_step, best_loss, cfg)

        # save best
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(best_ckpt, model, optimizer, scheduler,
                            epoch, global_step, best_loss, cfg)
            logger.info("  *** new best val_loss = %.4f ***", best_loss)

    writer.close()

    total_time = time.time() - train_start
    hours, rem = divmod(int(total_time), 3600)
    mins, secs = divmod(rem, 60)
    logger.info("Training complete. Best val loss: %.4f | Total time: %dh %dm %ds", best_loss, hours, mins, secs)
    logger.info("Best model saved to: %s", best_ckpt)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BERT4Rec on MovieLens 1M")

    # Data
    p.add_argument("--data_dir",    default="data")
    p.add_argument("--max_seq_len", type=int,   default=200)
    p.add_argument("--mask_prob",   type=float, default=0.2)
    p.add_argument("--num_workers", type=int,   default=0,
                   help="0 is safest on macOS / MPS")
    p.add_argument("--use_features", action="store_true",
                   help="Enable multi-feature embedding (genres + decade)")

    # Model
    p.add_argument("--hidden_size",             type=int,   default=256)
    p.add_argument("--num_hidden_layers",       type=int,   default=2)
    p.add_argument("--num_attention_heads",     type=int,   default=4)
    p.add_argument("--intermediate_size",       type=int,   default=1024)
    p.add_argument("--hidden_dropout_prob",     type=float, default=0.1)
    p.add_argument("--attention_probs_dropout", type=float, default=0.1)

    # Training
    p.add_argument("--epochs",          type=int,   default=200)
    p.add_argument("--batch_size",      type=int,   default=256)
    p.add_argument("--learning_rate",   type=float, default=1e-3)
    p.add_argument("--weight_decay",    type=float, default=1e-2)
    p.add_argument("--warmup_steps",    type=int,   default=100)
    p.add_argument("--lr_min_ratio",    type=float, default=0.1)
    p.add_argument("--grad_clip",       type=float, default=5.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--seed",            type=int,   default=42)

    # Infra
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--log_dir",        default="runs/bert4rec")
    p.add_argument("--resume",         default=None,
                   help="Path to a checkpoint to resume from")
    p.add_argument("--log_level",      default="INFO",
                   choices=["DEBUG", "INFO", "WARNING"])
    return p.parse_args()


if __name__ == "__main__":
    args  = _parse_args()
    log_level = args.log_level
    cfg   = vars(args)
    cfg.pop("log_level", None)

    logging.basicConfig(
        level   = log_level,
        format  = "%(asctime)s | %(levelname)s | %(message)s",
        datefmt = "%H:%M:%S",
    )

    # Reproducibility
    torch.manual_seed(cfg["seed"])

    train(cfg)
