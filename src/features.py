"""
features.py — Encoding functions for ClinVar SNV variants.

Provides one-hot encoding for:
  - Allele pairs (ref + alt) → 8-dimensional vector
  - Enriched features (allele + gene + chromosome) → 84-dimensional vector
  - Flanking sequences (FASTA context) → (SEQ_LENGTH, 4) matrix

All encodings are deterministic and reproducible.
"""

from pathlib import Path
from typing import Optional
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

NUCLEOTIDES = ['A', 'C', 'G', 'T']
VOCAB       = {n: i for i, n in enumerate(NUCLEOTIDES)}

TOP_N_GENES = 50

CHROMOSOMES = [
    '1','2','3','4','5','6','7','8','9','10',
    '11','12','13','14','15','16','17','18','19','20',
    '21','22','X','Y','MT'
]

CHROM_TO_IDX = {c: i for i, c in enumerate(CHROMOSOMES)}

# ── Allele encoding ───────────────────────────────────────────────────────────

def one_hot_base(base: str) -> list:
    """One-hot encode a single nucleotide as a 4-element list."""
    vector = [0, 0, 0, 0]
    if base in VOCAB:
        vector[VOCAB[base]] = 1
    return vector


def encode_allele(ref: str, alt: str) -> list:
    """
    Encode a SNV as a concatenated one-hot vector of length 8.
    Format: [ref_onehot (4), alt_onehot (4)]
    """
    return one_hot_base(ref) + one_hot_base(alt)


# ── Enriched feature encoding ─────────────────────────────────────────────────

def build_gene_index(gene_counts: dict) -> dict:
    """
    Build gene → index mapping from a frequency dictionary.
    Top N genes get indices 0..N-1; everything else maps to N (other bucket).
    """
    top_genes = sorted(gene_counts, key=gene_counts.get, reverse=True)[:TOP_N_GENES]
    return {gene: i for i, gene in enumerate(top_genes)}


def encode_enriched(
    ref: str,
    alt: str,
    gene: str,
    chrom: str,
    gene_to_idx: dict,
) -> list:
    """
    Encode a variant with allele + gene + chromosome context.

    Input dim: 8 (allele) + 51 (top-50 genes + other) + 25 (chromosomes) = 84

    Args:
        ref:         Reference allele (single base)
        alt:         Alternate allele (single base)
        gene:        Gene symbol (e.g. 'BRCA2')
        chrom:       Chromosome (e.g. '17', 'X', 'MT')
        gene_to_idx: Mapping from gene symbol to index (built via build_gene_index)
    """
    allele   = encode_allele(ref, alt)

    gene_vec = [0] * (TOP_N_GENES + 1)
    gene_vec[gene_to_idx.get(gene, TOP_N_GENES)] = 1

    chrom_vec = [0] * len(CHROMOSOMES)
    chrom_vec[CHROM_TO_IDX.get(str(chrom), 0)] = 1

    return allele + gene_vec + chrom_vec


# ── Sequence encoding ─────────────────────────────────────────────────────────

def encode_sequence(seq: str, length: int) -> np.ndarray:
    """
    One-hot encode a DNA sequence into a (length, 4) float32 matrix.
    Unknown bases (N, gaps) are encoded as all zeros.

    Args:
        seq:    DNA sequence string (uppercase)
        length: Expected sequence length (2 * FLANK + 1)
    """
    matrix = np.zeros((length, 4), dtype=np.float32)
    for i, base in enumerate(seq[:length]):
        if base in VOCAB:
            matrix[i, VOCAB[base]] = 1.0
    return matrix


def get_flanking_sequence(genome, chrom: str, position: int, flank: int) -> Optional[str]:
    """
    Extract flanking sequence around a variant position from a pyfaidx genome.

    Args:
        genome:   pyfaidx.Fasta object
        chrom:    Chromosome string (e.g. '17', 'X', 'MT')
        position: 1-based VCF position
        flank:    Number of bases either side of variant

    Returns:
        Uppercase sequence string of length 2*flank+1, or None if out of bounds.
    """
    chrom_name = 'chrM' if chrom == 'MT' else f'chr{chrom}'
    if chrom_name not in genome:
        return None
    pos_0based = int(position) - 1
    start = max(0, pos_0based - flank)
    end   = pos_0based + flank + 1
    return str(genome[chrom_name][start:end]).upper()