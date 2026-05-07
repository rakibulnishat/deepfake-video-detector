"""
train.py — Training loop for Deepfake Detection
Supports:
  - Mixed precision (AMP) for faster P100/T4 training
  - Combined loss: BCE + orthogonal + contrastive
  - Cosine LR scheduler with warmup
  - Checkpoint saving + resuming
  - Optional Weights & Biases logging
"""

import os
import time
import math
import yaml
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np


# ─── WARMUP + COSINE LR SCHEDULER ────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-6, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


# ─── METRICS TRACKER ─────────────────────────────────────────────────────────

class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.all_probs = []
        self.all_labels = []

    def update(self, loss: float, probs: torch.Tensor, labels: torch.Tensor):
        self.losses.append(loss)
        self.all_probs.extend(probs.detach().cpu().numpy().tolist())
        self.all_labels.extend(labels.detach().cpu().numpy().tolist())

    def compute(self) -> Dict[str, float]:
        avg_loss = np.mean(self.losses)
        probs = np.array(self.all_probs)
        labels = np.array(self.all_labels)

        auc = 0.0
        try:
            if len(np.unique(labels)) == 2:
                auc = roc_auc_score(labels, probs)
        except Exception:
            pass

        preds = (probs >= 0.5).astype(int)
        acc = (preds == labels).mean()

        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {
            "loss": avg_loss,
            "auc": auc,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }


# ─── CHECKPOINT HELPERS ──────────────────────────────────────────────────────

def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch: int,
    metrics: Dict, cfg: dict
):
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:02d}_auc{metrics['auc']:.4f}.pt"
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metrics": metrics,
        "config": cfg,
    }, ckpt_path)
    print(f"  ✓ Saved checkpoint: {ckpt_path.name}")
    return ckpt_path


def load_checkpoint(model, optimizer, scheduler, scaler, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    print(f"  ✓ Resumed from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


# ─── TRAIN ONE EPOCH ─────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    cfg: dict,
    device: str,
    epoch: int,
    wandb_run=None,
) -> Dict[str, float]:

    model.train()
    tracker = MetricsTracker()
    t_cfg = cfg["training"]
    log_interval = cfg["logging"]["log_interval"]

    bce_w = t_cfg["bce_weight"]
    orth_w = t_cfg["orthogonal_loss_weight"]
    cont_w = t_cfg["contrastive_loss_weight"]

    for step, batch in enumerate(loader):
        frames = batch["frames"].to(device, non_blocking=True)    # [B, T, C, H, W]
        labels = batch["label"].to(device, non_blocking=True)     # [B]

        optimizer.zero_grad()

        with autocast(enabled=(scaler is not None)):
            output = model(frames, labels)
            losses = output["losses"]
            total_loss = (
                bce_w  * losses["bce"]
              + orth_w * losses["orthogonal"]
              + cont_w * losses["contrastive"]
            )

        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), t_cfg["grad_clip"]
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), t_cfg["grad_clip"]
            )
            optimizer.step()

        tracker.update(
            total_loss.item(),
            output["probs"],
            labels,
        )

        if step % log_interval == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Step {step}/{len(loader)} | "
                f"Loss: {total_loss.item():.4f} | "
                f"BCE: {losses['bce'].item():.4f} | "
                f"Orth: {losses['orthogonal'].item():.4f} | "
                f"LR: {current_lr:.2e}"
            )

            if wandb_run:
                wandb_run.log({
                    "train/step_loss": total_loss.item(),
                    "train/bce_loss": losses["bce"].item(),
                    "train/orthogonal_loss": losses["orthogonal"].item(),
                    "train/contrastive_loss": losses["contrastive"].item(),
                    "train/lr": current_lr,
                    "epoch": epoch,
                })

    return tracker.compute()


# ─── VALIDATE ────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:

    model.eval()
    tracker = MetricsTracker()

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        output = model(frames, labels=None)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"], labels
        )
        tracker.update(loss.item(), output["probs"], labels)

    return tracker.compute()


# ─── MAIN TRAINING FUNCTION ──────────────────────────────────────────────────

def train(cfg: dict, model, train_loader, val_loader, device: str):
    """Full training loop. Called from the notebook."""
    t_cfg = cfg["training"]
    log_cfg = cfg["logging"]

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=t_cfg["learning_rate"],
        weight_decay=t_cfg["weight_decay"],
    )

    # ── Scheduler ──
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=t_cfg["warmup_epochs"],
        total_epochs=t_cfg["epochs"],
    )

    # ── AMP scaler ──
    scaler = GradScaler() if t_cfg["use_amp"] and torch.cuda.is_available() else None

    # ── Optional W&B ──
    wandb_run = None
    if log_cfg.get("wandb_api_key"):
        try:
            import wandb
            wandb.login(key=log_cfg["wandb_api_key"])
            wandb_run = wandb.init(
                project=log_cfg["wandb_project"],
                name=log_cfg["wandb_run_name"],
                config=cfg,
            )
            print("✓ Weights & Biases logging enabled.")
        except Exception as e:
            print(f"[WARNING] W&B init failed: {e}. Logging disabled.")

    # ── Training loop ──
    best_auc = 0.0
    best_ckpt = None

    print("=" * 60)
    print(f"Starting training for {t_cfg['epochs']} epochs")
    print(f"Device: {device}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print("=" * 60)

    for epoch in range(1, t_cfg["epochs"] + 1):
        epoch_start = time.time()
        print(f"\n── Epoch {epoch}/{t_cfg['epochs']} ──")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler,
            cfg, device, epoch, wandb_run
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Scheduler step
        scheduler.step()

        elapsed = time.time() - epoch_start
        print(
            f"\n  TRAIN → Loss: {train_metrics['loss']:.4f} | "
            f"AUC: {train_metrics['auc']:.4f} | F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"  VAL   → Loss: {val_metrics['loss']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"Acc: {val_metrics['acc']:.4f}"
        )
        print(f"  Time: {elapsed:.1f}s")

        if wandb_run:
            wandb_run.log({
                "epoch": epoch,
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_auc": train_metrics["auc"],
                "train/epoch_f1": train_metrics["f1"],
                "val/loss": val_metrics["loss"],
                "val/auc": val_metrics["auc"],
                "val/f1": val_metrics["f1"],
                "val/acc": val_metrics["acc"],
            })

        # Save checkpoint every N epochs
        if epoch % t_cfg["save_every"] == 0:
            ckpt_path = save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_metrics, cfg
            )
            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_ckpt = ckpt_path
                print(f"  ★ New best AUC: {best_auc:.4f}")

    print("\n" + "=" * 60)
    print(f"Training complete. Best Val AUC: {best_auc:.4f}")
    if best_ckpt:
        print(f"Best checkpoint: {best_ckpt}")
    print("=" * 60)

    if wandb_run:
        wandb_run.finish()

    return best_ckpt
