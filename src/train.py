"""
train.py — Training loop for ClinVar variant classifiers.

Supports all three model types (VariantMLP, EnrichedMLP, VariantCNN).
W&B logging is built in — every run is tracked automatically.

Usage:
    from src.train import train_model
    history, best_state = train_model(model, train_loader, val_loader, config)

Library candidate: reusable training loop for any PyTorch binary
classification task with W&B integration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Optional
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ── Epoch functions ───────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Returns:
        (avg_loss, auc) for the epoch.
    """
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimiser.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimiser.step()

        total_loss += loss.item()
        all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc      = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Run one evaluation epoch (no gradient updates).

    Returns:
        (avg_loss, auc) for the epoch.
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)

            total_loss += loss.item()
            all_preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    auc      = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc


# ── Main training loop ────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    run_config: dict,
    save_path: Optional[str] = None,
    wandb_project: str = "clinvar-classifier",
) -> tuple[dict, dict]:
    """
    Full training loop with W&B logging and early stopping on val AUC plateau.

    Args:
        model:          PyTorch model to train
        train_loader:   Training DataLoader
        val_loader:     Validation DataLoader
        run_config:     Dict of hyperparameters and metadata — logged to W&B.
                        Required keys: device, learning_rate, max_epochs,
                        pos_weight, model_name, hypothesis
        save_path:      Path to save best model state (.pt file). Optional.
        wandb_project:  W&B project name.

    Returns:
        history:    Dict of train/val loss and AUC per epoch.
        best_state: State dict of best model (highest val AUC).

    Example run_config:
        {
            "model_name":    "VariantCNN",
            "hypothesis":    "Sequence context generalises across gene groups",
            "encoding":      "flanking_sequence_101bp",
            "flank":         50,
            "learning_rate": 1e-3,
            "batch_size":    32,
            "max_epochs":    50,
            "dropout":       0.3,
            "pos_weight":    4.61,
            "device":        "cuda",
            "dataset":       "ClinVar_GRCh38_germline_2star",
            "n_train":       239277,
            "n_val":         34183,
            "random_seed":   42,
        }
    """
    device      = torch.device(run_config["device"])
    max_epochs  = run_config["max_epochs"]
    lr          = run_config["learning_rate"]
    pos_weight  = torch.tensor([run_config["pos_weight"]], dtype=torch.float32).to(device)

    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_auc": [],
        "val_loss":   [], "val_auc":   [],
    }

    best_val_auc  = 0.0
    best_state    = None

    # ── W&B init ──────────────────────────────────────────────────────────────
    use_wandb = WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(
                project = wandb_project,
                name    = run_config.get("model_name", "run"),
                tags    = run_config.get("tags", []),
                config  = run_config,
            )
        except Exception as e:
            print(f"W&B init failed: {e} — continuing without logging.")
            use_wandb = False

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining {run_config.get('model_name', 'model')} "
          f"for {max_epochs} epochs on {device}")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train AUC':>12} "
          f"{'Val Loss':>10} {'Val AUC':>10}")
    print("-" * 56)

    for epoch in range(1, max_epochs + 1):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimiser, device
        )
        val_loss, val_auc = evaluate_epoch(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_auc"].append(train_auc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        # W&B logging
        if use_wandb:
            wandb.log({
                "epoch":      epoch,
                "train/loss": train_loss,
                "train/auc":  train_auc,
                "val/loss":   val_loss,
                "val/auc":    val_auc,
            })

        print(f"{epoch:>6} {train_loss:>12.4f} {train_auc:>12.4f} "
              f"{val_loss:>10.4f} {val_auc:>10.4f}")

    print("-" * 56)
    print(f"Best validation AUC: {best_val_auc:.4f}")

    # ── Save best model ───────────────────────────────────────────────────────
    if save_path is not None:
        torch.save(best_state, save_path)
        print(f"Best model saved to {save_path}")

    # ── W&B finish ────────────────────────────────────────────────────────────
    if use_wandb:
        wandb.run.summary["best_val_auc"] = best_val_auc
        wandb.finish()

    return history, best_state