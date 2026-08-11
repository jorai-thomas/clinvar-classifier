"""
evaluate.py — Evaluation functions for ClinVar variant classifiers.

Provides:
  - get_predictions()     — run inference on a test set
  - per_group_auc()       — AUC split by gene representation group
  - plot_evaluation()     — ROC, PR, calibration, per-group AUC
  - plot_training_curves()— loss and AUC curves from training history
  - save_predictions()    — structured JSON output for downstream use

Library candidate: reusable evaluation suite for any binary genomic
classification task with equity auditing.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Inference ─────────────────────────────────────────────────────────────────

def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader and return predictions and labels.

    Returns:
        preds:  Sigmoid probabilities (float array)
        labels: Ground truth binary labels (int array)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits  = model(X_batch)
            probs   = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(y_batch.numpy())

    return np.array(all_preds), np.array(all_labels)


# ── Per-group AUC ─────────────────────────────────────────────────────────────

def per_group_auc(
    preds: np.ndarray,
    labels: np.ndarray,
    gene_labels: np.ndarray,
    top_genes: list,
) -> dict:
    """
    Compute AUC split by gene representation group.

    This is the core equity metric for this project. A large gap between
    top-50 and other genes indicates the model has inherited ClinVar's
    ascertainment bias.

    Args:
        preds:       Model probability predictions
        labels:      Ground truth binary labels
        gene_labels: Gene symbol for each test variant
        top_genes:   List of top-N gene symbols

    Returns:
        Dict with keys: overall, top50, other, gap
    """
    top50_mask = np.array([g in top_genes for g in gene_labels])
    other_mask = ~top50_mask

    auc_overall = roc_auc_score(labels, preds)
    auc_top50   = roc_auc_score(labels[top50_mask], preds[top50_mask])
    auc_other   = roc_auc_score(labels[other_mask], preds[other_mask])

    return {
        "overall":        round(auc_overall, 4),
        "top50_genes":    round(auc_top50, 4),
        "other_genes":    round(auc_other, 4),
        "gap":            round(auc_top50 - auc_other, 4),
        "n_top50":        int(top50_mask.sum()),
        "n_other":        int(other_mask.sum()),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_evaluation(
    models: dict,
    gene_labels: np.ndarray,
    top_genes: list,
    save_path: Path,
) -> None:
    """
    Generate full evaluation summary — ROC, PR, calibration, per-group AUC.

    Args:
        models:      Dict of {name: (labels, preds, colour)}
        gene_labels: Gene symbol for each test variant
        top_genes:   List of top-N gene symbols
        save_path:   Path to save the figure (.png)
    """
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── ROC curves ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for name, (y, preds, colour) in models.items():
        fpr, tpr, _ = roc_curve(y, preds)
        auc = roc_auc_score(y, preds)
        ax1.plot(fpr, tpr, color=colour, label=f"{name} (AUC={auc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves')
    ax1.legend(fontsize=9)

    # ── Precision-Recall curves ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for name, (y, preds, colour) in models.items():
        prec, rec, _ = precision_recall_curve(y, preds)
        ap = average_precision_score(y, preds)
        ax2.plot(rec, prec, color=colour, label=f"{name} (AP={ap:.3f})")
    baseline_pr = list(models.values())[0][0].mean()
    ax2.axhline(y=baseline_pr, color='k', linestyle='--',
                label=f'Random (AP={baseline_pr:.3f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curves')
    ax2.legend(fontsize=9)

    # ── Calibration curves ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for name, (y, preds, colour) in models.items():
        fraction_pos, mean_pred = calibration_curve(y, preds, n_bins=10)
        ax3.plot(mean_pred, fraction_pos, color=colour, marker='o', label=name)
    ax3.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    ax3.set_xlabel('Mean Predicted Probability')
    ax3.set_ylabel('Fraction of Positives')
    ax3.set_title('Calibration Curves')
    ax3.legend(fontsize=9)

    # ── Per-group AUC ─────────────────────────────────────────────────────────
    ax4  = fig.add_subplot(gs[1, 1])
    names      = list(models.keys())
    top50_aucs = []
    other_aucs = []

    for name, (y, preds, _) in models.items():
        result = per_group_auc(preds, y, gene_labels, top_genes)
        top50_aucs.append(result["top50_genes"])
        other_aucs.append(result["other_genes"])

    x     = np.arange(len(names))
    width = 0.3
    bars1 = ax4.bar(x - width/2, top50_aucs, width,
                    label='Top-50 genes', color='steelblue')
    bars2 = ax4.bar(x + width/2, other_aucs, width,
                    label='"Other" genes', color='coral')
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, fontsize=9)
    ax4.set_ylabel('AUC')
    ax4.set_title('Per-Group AUC: Top-50 vs Other Genes')
    ax4.legend(fontsize=9)
    ax4.set_ylim(0.5, 0.9)

    for bar in bars1:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f'{bar.get_height():.3f}',
                 ha='center', va='bottom', fontsize=8)

    plt.suptitle('ClinVar Variant Classifier — Model Evaluation',
                 fontsize=14, fontweight='bold')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Evaluation figure saved to {save_path}")


def plot_training_curves(
    history: dict,
    model_name: str,
    baseline_auc: float,
    save_path: Path,
) -> None:
    """
    Plot loss and AUC curves from a training history dict.

    Args:
        history:      Dict with keys train_loss, val_loss, train_auc, val_auc
        model_name:   Label for the plot title
        baseline_auc: Reference line on AUC plot (e.g. previous model's AUC)
        save_path:    Path to save the figure (.png)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label='Train')
    ax1.plot(epochs, history["val_loss"],   label='Validation')
    ax1.set_title(f'{model_name} — Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('BCE Loss')
    ax1.legend()

    ax2.plot(epochs, history["train_auc"], label='Train')
    ax2.plot(epochs, history["val_auc"],   label='Validation')
    ax2.axhline(y=baseline_auc, color='r', linestyle='--',
                label=f'Baseline ({baseline_auc:.2f})')
    ax2.axhline(y=0.5, color='gray', linestyle=':', label='Random')
    ax2.set_title(f'{model_name} — AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('ROC-AUC')
    ax2.legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


# ── Structured JSON output ────────────────────────────────────────────────────

def save_predictions(
    preds: np.ndarray,
    labels: np.ndarray,
    gene_labels: np.ndarray,
    top_genes: list,
    model_name: str,
    save_path: Path,
) -> None:
    """
    Save model predictions as structured JSON for downstream use.

    Output format is agent-ready — machine readable, not just printed metrics.
    Each record contains the prediction, label, gene, and group membership.

    Args:
        preds:       Model probability predictions
        labels:      Ground truth binary labels
        gene_labels: Gene symbol for each variant
        top_genes:   List of top-N gene symbols
        model_name:  Model identifier string
        save_path:   Path to save JSON file
    """
    group_results = per_group_auc(preds, labels, gene_labels, top_genes)

    output = {
        "model":        model_name,
        "n_variants":   len(preds),
        "metrics": {
            "auc_overall":     group_results["overall"],
            "auc_top50_genes": group_results["top50_genes"],
            "auc_other_genes": group_results["other_genes"],
            "auc_gap":         group_results["gap"],
            "average_precision": round(
                average_precision_score(labels, preds), 4
            ),
        },
        "group_counts": {
            "top50_genes": group_results["n_top50"],
            "other_genes": group_results["n_other"],
        },
        "predictions": [
            {
                "index":       int(i),
                "probability": round(float(preds[i]), 4),
                "label":       int(labels[i]),
                "gene":        str(gene_labels[i]),
                "gene_group":  "top50" if gene_labels[i] in top_genes else "other",
            }
            for i in range(len(preds))
        ],
    }

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Predictions saved to {save_path} ({len(preds):,} variants)")