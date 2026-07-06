from pathlib import Path

# ── Root ──────────────────────────────────────────────────────────────────────
# Change this if working on a different machine — everything else resolves from it
ROOT = Path(__file__).parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR     = ROOT / "outputs"
FIGURES_DIR     = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# ── ClinVar source files (populated after download) ───────────────────────────
VARIANT_SUMMARY = RAW_DIR / "variant_summary.txt.gz"
CLINVAR_VCF     = RAW_DIR / "clinvar.vcf.gz"

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    "Pathogenic":           1,
    "Likely pathogenic":    1,
    "Benign":               0,
    "Likely benign":        0,
}

# ── Filtering constants ────────────────────────────────────────────────────────
MIN_REVIEW_STARS    = 2       # Minimum ClinVar gold stars to include a variant
EXCLUDE_LABELS      = [       # Labels to drop entirely
    "Uncertain significance",
    "Conflicting interpretations of pathogenicity",
    "not provided",
    "other",
]

# ── Model / training hyperparameters (placeholders for now) ───────────────────
RANDOM_SEED     = 42
TEST_SIZE       = 0.2
VAL_SIZE        = 0.1
BATCH_SIZE      = 32
LEARNING_RATE   = 1e-3
MAX_EPOCHS      = 50

# ── Sequence parameters ───────────────────────────────────────────────────────
SEQ_LENGTH      = 100         # Flanking sequence length (each side of variant)
NUCLEOTIDES     = ["A", "C", "G", "T", "N"]

# ── Device configuration ──────────────────────────────────────────────────────
import torch

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')        # Windows with NVIDIA GPU
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')         # Apple Silicon Mac
else:
    DEVICE = torch.device('cpu')         # Fallback