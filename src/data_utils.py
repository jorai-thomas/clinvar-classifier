"""
data_utils.py — Data loading and filtering for ClinVar variant data.

Provides:
  - load_filtered()     — load the processed filtered TSV
  - filter_variants()   — apply quality filters to raw variant summary
  - VariantDataset      — PyTorch Dataset for allele/enriched features
  - SeqDataset          — PyTorch Dataset for flanking sequence features
  - make_splits()       — stratified train/val/test split

Library candidate: VariantDataset and SeqDataset are reusable for any
tabular or sequence-based genomic classification task.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Optional


# ── Label and filter constants ────────────────────────────────────────────────

LABEL_MAP = {
    "Pathogenic":                   1,
    "Likely pathogenic":            1,
    "Pathogenic/Likely pathogenic": 1,
    "Benign":                       0,
    "Likely benign":                0,
    "Benign/Likely benign":         0,
}

KEEP_REVIEW = {
    "practice guideline",
    "reviewed by expert panel",
    "criteria provided, multiple submitters, no conflicts",
}

KEEP_ASSEMBLY = "GRCh38"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_filtered(processed_dir: Path) -> pd.DataFrame:
    """
    Load the pre-filtered ClinVar TSV produced by 01_eda.ipynb.

    Args:
        processed_dir: Path to data/processed/

    Returns:
        DataFrame with 383k high-confidence SNVs.
    """
    path = processed_dir / "clinvar_filtered.tsv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Filtered data not found at {path}. "
            "Run notebooks/01_eda.ipynb first."
        )
    return pd.read_csv(path, sep='\t', low_memory=False)


def filter_variants(raw_path: Path, chunk_size: int = 50_000) -> pd.DataFrame:
    """
    Apply quality filters to raw ClinVar variant_summary.txt.gz.

    Filters:
      - Assembly == GRCh38 (removes GRCh37 duplicates)
      - Origin contains 'germline' (removes somatic variants)
      - ReviewStatus in KEEP_REVIEW (⭐⭐+ only)
      - ClinicalSignificance in LABEL_MAP (unambiguous labels only)

    Processes in chunks to avoid loading 2-3GB into memory.

    Args:
        raw_path:   Path to variant_summary.txt.gz
        chunk_size: Rows per chunk (default 50k)

    Returns:
        Filtered DataFrame with Label column added.
    """
    chunks = []
    reader = pd.read_csv(raw_path, sep='\t', chunksize=chunk_size, low_memory=False)

    for chunk in reader:
        chunk = chunk[chunk['Assembly'] == KEEP_ASSEMBLY]
        chunk = chunk[chunk['Origin'].str.contains('germline', na=False)]
        chunk = chunk[chunk['ReviewStatus'].isin(KEEP_REVIEW)]
        chunk = chunk[chunk['ClinicalSignificance'].isin(LABEL_MAP.keys())]
        chunk['Label'] = chunk['ClinicalSignificance'].map(LABEL_MAP)
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def filter_snvs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to single nucleotide variants only.

    Drops:
      - Non-SNV variant types
      - Rows with allele length > 1
      - Rows with 'na' allele values

    Args:
        df: Filtered ClinVar DataFrame

    Returns:
        SNV-only DataFrame, index reset.
    """
    df = df[df['Type'] == 'single nucleotide variant'].copy()
    df = df[df['ReferenceAlleleVCF'].str.len() == 1]
    df = df[df['AlternateAlleleVCF'].str.len() == 1]
    df = df[df['ReferenceAlleleVCF'] != 'na']
    df = df[df['AlternateAlleleVCF'] != 'na']
    return df.reset_index(drop=True)


# ── PyTorch Datasets ──────────────────────────────────────────────────────────

class VariantDataset(Dataset):
    """
    PyTorch Dataset for allele or enriched feature vectors.

    Used with VariantMLP and EnrichedMLP.
    Input X should be a numpy array of shape (n_variants, feature_dim).

    Library candidate: reusable for any tabular genomic classification task.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


class SeqDataset(Dataset):
    """
    PyTorch Dataset for flanking sequence matrices.

    Used with VariantCNN and VariantCNNWide.
    Input X should be a numpy array of shape (n_variants, SEQ_LENGTH, 4).
    Automatically transposes to (n_variants, 4, SEQ_LENGTH) for Conv1d.

    Library candidate: reusable for any 1D genomic sequence classification task.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # Transpose (n, length, 4) → (n, 4, length) for Conv1d
        self.X = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        return self.X[idx], self.y[idx]


# ── Splits and DataLoaders ────────────────────────────────────────────────────

def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_seed: int = 42,
) -> tuple:
    """
    Stratified train/val/test split preserving class ratio across all splits.

    Args:
        X:           Feature array
        y:           Label array
        test_size:   Fraction for test set (default 0.2)
        val_size:    Fraction for validation set (default 0.1)
        random_seed: Random seed for reproducibility

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size / (1 - test_size),
        random_state=random_seed,
        stratify=y_train_val,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def make_loaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    dataset_class,
    batch_size: int = 32,
) -> tuple:
    """
    Build train, val, and test DataLoaders from split arrays.

    Args:
        dataset_class: VariantDataset or SeqDataset
        batch_size:    Batch size (default 32)

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        dataset_class(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        dataset_class(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset_class(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader