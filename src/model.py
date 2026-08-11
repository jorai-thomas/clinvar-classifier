"""
model.py — PyTorch model definitions for ClinVar variant classification.

Three architectures in order of complexity:

  VariantMLP     — baseline, allele-only input (8-dim)
  EnrichedMLP    — gene + chromosome context (84-dim)
  VariantCNN     — flanking sequence context (4 x SEQ_LENGTH)

All models output a single logit for binary classification.
BCEWithLogitsLoss is the intended loss function.

Library candidate: VariantMLP and VariantCNN are reusable starting
points for any binary genomic classification task.
"""

import torch
import torch.nn as nn


# ── Baseline MLP ──────────────────────────────────────────────────────────────

class VariantMLP(nn.Module):
    """
    Baseline MLP for ClinVar SNV pathogenicity classification.

    Input:  8-dimensional one-hot encoded variant (ref + alt alleles)
    Output: single logit for binary classification

    Architecture: 3 fully connected layers with ReLU and dropout.
    Deliberately simple — establishes baseline before richer encodings.

    Result: AUC 0.59 — confirms allele type alone is insufficient.
    Per-group gap: +0.08 (top-50 genes vs understudied genes).
    """

    def __init__(self, input_dim: int = 8, hidden_dims: list = [64, 32], dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# ── Enriched MLP ──────────────────────────────────────────────────────────────

class EnrichedMLP(nn.Module):
    """
    MLP with gene and chromosome context for ClinVar SNV classification.

    Input:  84-dimensional vector
              8  — one-hot allele (ref + alt)
              51 — one-hot gene (top-50 + other bucket)
              25 — one-hot chromosome
    Output: single logit for binary classification

    Wider than VariantMLP to handle richer input.

    Result: AUC 0.67 overall — but AUC gap +0.22 between top-50 and
    understudied genes. Demonstrates gene-identity encoding inherits
    ClinVar's ascertainment bias.
    """

    def __init__(self, input_dim: int = 84, hidden_dims: list = [256, 128, 64], dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[2], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


# ── CNN ───────────────────────────────────────────────────────────────────────

class VariantCNN(nn.Module):
    """
    1D CNN for ClinVar SNV pathogenicity classification using flanking sequence.

    Input:  (batch, 4, SEQ_LENGTH) — one-hot encoded flanking sequence
              4 channels (ACGT), SEQ_LENGTH positions (2 * FLANK + 1)
    Output: single logit for binary classification

    Architecture: convolutional layers detect local sequence motifs
    (splice sites, conserved regions), global average pooling collapses
    to fixed-size vector, classifier head outputs logit.

    Does not use gene identity — learns transferable biological rules
    that generalise equitably across gene representation groups.

    Result (FLANK=50, 101bp): AUC 0.73, per-group gap -0.02.
    Result (FLANK=150, 301bp): AUC 0.72, per-group gap -0.01.
    Wider context did not improve performance — suggests attention-based
    architectures (DNABERT) needed for longer-range dependencies.

    Library candidate: reusable for any binary sequence classification task.
    """

    def __init__(self, n_filters_1: int = 64, n_filters_2: int = 128, dropout: float = 0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, n_filters_1, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(n_filters_2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x).squeeze(1)


# ── Wide CNN ──────────────────────────────────────────────────────────────────

class VariantCNNWide(nn.Module):
    """
    Deeper 1D CNN for wider flanking sequence context (FLANK=150, 301bp).

    Same principle as VariantCNN but with a third convolutional layer
    to handle the longer input. Use with SEQ_LENGTH=301.

    Result: AUC 0.72 — no meaningful improvement over FLANK=50 CNN,
    confirming simple convolution hits ceiling regardless of context width.
    """

    def __init__(self, n_filters_1: int = 64, n_filters_2: int = 128, n_filters_3: int = 256, dropout: float = 0.3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv1d(4, n_filters_1, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(n_filters_2, n_filters_3, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(n_filters_3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x).squeeze(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pos_weight(y_train) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss from training labels.
    Handles class imbalance by weighting the minority class (Pathogenic).

    Args:
        y_train: numpy array of binary labels (0=Benign, 1=Pathogenic)

    Returns:
        Scalar tensor for use as pos_weight in BCEWithLogitsLoss.
    """
    import numpy as np
    counts = np.bincount(y_train)
    return torch.tensor([counts[0] / counts[1]], dtype=torch.float32)