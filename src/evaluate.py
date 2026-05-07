"""
evaluate.py — Full evaluation suite for the trained deepfake detector.
Computes:
  - AUC-ROC (primary metric per research plan)
  - F1-score at optimal threshold
  - Equal Error Rate (EER)
  - Precision / Recall / Accuracy
  - Saves prediction CSV for analysis
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ─── EQUAL ERROR RATE ────────────────────────────────────────────────────────

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the threshold at which FAR == FRR.
    Used in biometric/anti-spoofing literature.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    # EER is at the point where FPR ≈ FNR
    eer_threshold_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    eer_threshold = thresholds[eer_threshold_idx]
    return float(eer), float(eer_threshold)


# ─── OPTIMAL F1 THRESHOLD ────────────────────────────────────────────────────

def optimal_f1_threshold(
    labels: np.ndarray, probs: np.ndarray
) -> Tuple[float, float]:
    """Find the probability threshold that maximises F1 on this dataset."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])   # last threshold is always 1.0
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ─── PLOT HELPERS ────────────────────────────────────────────────────────────

def plot_roc_curve(
    labels: np.ndarray, probs: np.ndarray,
    auc: float, save_path: str
):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#E84855", lw=2,
             label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Deepfake Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ ROC curve saved: {save_path}")


def plot_confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, save_path: str
):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    classes = ["REAL", "FAKE"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


# ─── MAIN EVALUATE FUNCTION ──────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    results_dir: str = "/kaggle/working/results",
    split_name: str = "val",
) -> Dict[str, float]:
    """
    Run full evaluation on a DataLoader.
    Returns a dict of all metrics, saves plots and CSV.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    all_probs: List[float] = []
    all_labels: List[int] = []
    all_paths: List[str] = []

    print(f"\nEvaluating on {split_name} set...")

    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"]
        paths = batch["path"]

        output = model(frames, labels=None)
        probs = output["probs"].cpu().numpy().tolist()

        all_probs.extend(probs)
        all_labels.extend(labels.numpy().tolist())
        all_paths.extend(paths)

    # Convert to numpy
    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # ── AUC-ROC ──────────────────────────────────────
    auc = roc_auc_score(labels, probs)

    # ── EER ──────────────────────────────────────────
    eer, eer_threshold = compute_eer(labels, probs)

    # ── Optimal F1 ────────────────────────────────────
    best_threshold, best_f1 = optimal_f1_threshold(labels, probs)
    preds = (probs >= best_threshold).astype(int)
    precision = (
        ((preds == 1) & (labels == 1)).sum() /
        (preds == 1).sum().clip(min=1)
    )
    recall = (
        ((preds == 1) & (labels == 1)).sum() /
        (labels == 1).sum().clip(min=1)
    )
    acc = (preds == labels).mean()

    # ── Average Precision ─────────────────────────────
    ap = average_precision_score(labels, probs)

    # ── Print results ─────────────────────────────────
    print("\n" + "=" * 55)
    print(f"  EVALUATION RESULTS ({split_name.upper()})")
    print("=" * 55)
    print(f"  AUC-ROC          : {auc:.4f}  ← PRIMARY METRIC")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Equal Error Rate : {eer:.4f}  (threshold: {eer_threshold:.3f})")
    print(f"  Best F1-score    : {best_f1:.4f} (threshold: {best_threshold:.3f})")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  Accuracy         : {acc:.4f}")
    print("=" * 55)

    metrics = {
        "auc": float(auc),
        "average_precision": float(ap),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "f1": float(best_f1),
        "f1_threshold": float(best_threshold),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(acc),
    }

    # ── Save metrics JSON ─────────────────────────────
    metrics_path = results_dir / f"{split_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  ✓ Metrics saved: {metrics_path}")

    # ── Save predictions CSV ──────────────────────────
    df = pd.DataFrame({
        "path": all_paths,
        "true_label": all_labels,
        "pred_prob": all_probs,
        "pred_label": (probs >= best_threshold).astype(int).tolist(),
    })
    csv_path = results_dir / f"{split_name}_predictions.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Predictions CSV: {csv_path}")

    # ── Save plots ────────────────────────────────────
    plot_roc_curve(labels, probs, auc,
                   str(results_dir / f"{split_name}_roc_curve.png"))
    plot_confusion_matrix(labels, preds,
                          str(results_dir / f"{split_name}_confusion_matrix.png"))

    return metrics


# ─── SINGLE VIDEO INFERENCE ──────────────────────────────────────────────────

@torch.no_grad()
def predict_video(
    model: torch.nn.Module,
    video_path: str,
    cfg: dict,
    device: str,
) -> Dict:
    """
    Run inference on a single video file.
    Returns prediction dict with probability and label.
    """
    from dataset import sample_frames_from_video, FaceExtractor, get_val_transforms

    face_size = cfg["data"]["face_size"]
    num_frames = cfg["data"]["frames_per_video"]
    transform = get_val_transforms(face_size)
    face_extractor = FaceExtractor(face_size=face_size, device=device)

    frames = sample_frames_from_video(
        video_path, num_frames=num_frames,
        face_extractor=face_extractor, face_size=face_size
    )

    if frames is None:
        return {"error": "Could not load video", "path": video_path}

    frame_tensors = torch.stack([transform(f) for f in frames], dim=0)
    frame_tensors = frame_tensors.unsqueeze(0).to(device)   # [1, T, C, H, W]

    model.eval()
    output = model(frame_tensors, labels=None)
    prob = output["probs"].item()

    return {
        "path": video_path,
        "fake_probability": round(prob, 4),
        "prediction": "FAKE" if prob >= 0.5 else "REAL",
        "confidence": round(max(prob, 1 - prob), 4),
    }
