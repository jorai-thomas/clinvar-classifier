"""
app.py — ClinVar Variant Classifier — Live Demo

Streamlit app for interactive pathogenicity prediction.
Uses the VariantCNN model with flanking sequences fetched
in real time from NCBI Entrez API.

Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
import torch
import sys
from pathlib import Path
from Bio import Entrez, SeqIO

sys.path.append(str(Path(__file__).parent))
from src.model import VariantCNN
from src.features import encode_sequence

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ClinVar Variant Classifier",
    page_icon="🧬",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
FLANK = 50
SEQ_LENGTH = 2 * FLANK + 1

# GRCh38 chromosome accession numbers for Entrez
CHROM_ACCESSIONS = {
    "1":  "NC_000001.11", "2":  "NC_000002.12", "3":  "NC_000003.12",
    "4":  "NC_000004.12", "5":  "NC_000005.10", "6":  "NC_000006.12",
    "7":  "NC_000007.14", "8":  "NC_000008.11", "9":  "NC_000009.12",
    "10": "NC_000010.11", "11": "NC_000011.10", "12": "NC_000012.12",
    "13": "NC_000013.11", "14": "NC_000014.9",  "15": "NC_000015.10",
    "16": "NC_000016.10", "17": "NC_000017.11", "18": "NC_000018.10",
    "19": "NC_000019.10", "20": "NC_000020.11", "21": "NC_000021.9",
    "22": "NC_000022.11", "X":  "NC_000023.11", "Y":  "NC_000024.10",
    "MT": "NC_012920.1",
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load CNN model — cached so it only loads once."""
    model = VariantCNN()
    weights_path = Path("outputs/cnn_best.pt")
    if not weights_path.exists():
        st.error("Model weights not found. Ensure outputs/cnn_best.pt exists.")
        st.stop()
    model.load_state_dict(
        torch.load(weights_path, map_location=torch.device('cpu'))
    )
    model.eval()
    return model

# ── Entrez sequence fetcher ───────────────────────────────────────────────────
def fetch_flanking_sequence(chrom: str, position: int, flank: int = FLANK):
    """
    Fetch flanking sequence from NCBI Entrez in real time.

    Args:
        chrom:    Chromosome string (e.g. '17', 'X')
        position: 1-based VCF position
        flank:    Bases either side of variant

    Returns:
        Uppercase sequence string of length 2*flank+1, or None on error.
    """
    accession = CHROM_ACCESSIONS.get(str(chrom))
    if accession is None:
        return None

    start = max(1, position - flank)
    end   = position + flank

    try:
        Entrez.email = "demo@clinvar-classifier.com"
        handle = Entrez.efetch(
            db       = "nuccore",
            id       = accession,
            rettype  = "fasta",
            retmode  = "text",
            seq_start= start,
            seq_stop = end,
        )
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return str(record.seq).upper()
    except Exception as e:
        st.error(f"Entrez API error: {e}")
        return None

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, seq: str, alt: str, position: int, flank: int = FLANK) -> dict:
    """
    Run CNN inference on a flanking sequence.

    Substitutes the alternate allele at the centre position before encoding,
    so the model sees the mutant sequence rather than the reference.
    """
    # Substitute alt allele at centre position
    seq_list = list(seq)
    centre   = min(flank, len(seq_list) - 1)
    seq_list[centre] = alt.upper()
    seq_mut  = ''.join(seq_list)

    # Encode and predict
    encoded = encode_sequence(seq_mut, SEQ_LENGTH)          # (101, 4)
    tensor  = torch.tensor(encoded).unsqueeze(0).permute(0, 2, 1)  # (1, 4, 101)

    with torch.no_grad():
        logit = model(tensor)
        prob  = torch.sigmoid(logit).item()

    return {
        "probability":   round(prob, 4),
        "prediction":    "Pathogenic" if prob >= 0.5 else "Benign",
        "confidence":    round(max(prob, 1 - prob) * 100, 1),
        "sequence":      seq_mut,
    }

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧬 ClinVar Variant Classifier")
st.markdown(
    "Predict whether a genetic variant is **Pathogenic** or **Benign** "
    "using a 1D CNN trained on 341,825 ClinVar SNVs. "
    "Flanking sequence context is fetched in real time from NCBI."
)

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    chrom = st.selectbox(
        "Chromosome",
        options=list(CHROM_ACCESSIONS.keys()),
        index=16,  # default chr17
    )
    position = st.number_input(
        "Position (GRCh38, 1-based)",
        min_value=1,
        value=43045629,
        help="e.g. 43045629 — a known BRCA1 position"
    )

with col2:
    ref = st.selectbox("Reference Allele", options=["A", "C", "G", "T"], index=1)
    alt = st.selectbox("Alternate Allele", options=["A", "C", "G", "T"], index=3)

# Example variants
with st.expander("📖 Understanding the predictions — read this first"):
    st.markdown("""
### Try an example variant
| Gene | Chr | Position | Ref | Alt | Expected |
|---|---|---|---|---|---|
| BRCA1 | 17 | 43045629 | C | T | Pathogenic |
| BRCA2 | 13 | 32316508 | A | T | Pathogenic |
| TP53  | 17 | 7674220  | G | A | Pathogenic |

*Positions are GRCh38. Copy values into the fields above.*

---

### Why some known pathogenic variants predict with low confidence

This CNN encodes **sequence context only** — it has no knowledge of which
gene a variant falls in. Some well-known pathogenic variants (e.g. in BRCA1)
may be predicted with lower confidence because their pathogenicity depends
partly on gene function, not just local sequence patterns.

**This is a deliberate design choice, not a bug.**

A model that gets BRCA1 right by memorising "BRCA1 variants tend to be
pathogenic" would fail on an equivalent variant in an understudied gene
from an underrepresented population — precisely the failure mode this
project is designed to expose. The enriched MLP (which does use gene
identity) achieves AUC 0.83 on well-studied genes but only 0.61 on
understudied ones — a 0.22 gap that reflects database bias, not biology.
This CNN achieves AUC 0.73 with a near-zero equity gap (−0.02), performing
consistently across the genome regardless of how well-studied a gene is.

The model is being honest: *"based on this sequence context alone, without
knowing this is BRCA1, I am not confident this is pathogenic."* That
honesty is clinically safer than false confidence built on gene identity.

### What comes next

The Year 3 dissertation addresses this directly. DNABERT — a transformer
pretrained on the full human genome — encodes up to 512bp of sequence
context versus this model's 101bp. That richer representation captures
enough biological signal (splice sites, regulatory motifs, conservation
patterns) to infer functional importance from sequence alone, without
relying on gene identity. The goal is a model that gets BRCA1 right
*and* gets an equivalent variant in an understudied gene right — not
because it knows which gene it's looking at, but because it understands
the sequence biology deeply enough not to need that shortcut.
""")
    st.caption("Positions are GRCh38. Copy values into the fields above.")

st.divider()

# Classify button
if st.button("🔬 Classify Variant", type="primary", use_container_width=True):

    if ref == alt:
        st.warning("Reference and alternate alleles must be different.")
    else:
        with st.spinner("Fetching flanking sequence from NCBI..."):
            seq = fetch_flanking_sequence(chrom, position)

        if seq is None:
            st.error("Could not fetch sequence from NCBI. Please try again.")
        elif len(seq) < SEQ_LENGTH:
            st.warning(
                f"Sequence too short ({len(seq)}bp) — variant may be near "
                "chromosome boundary. Try a different position."
            )
        else:
            with st.spinner("Running CNN inference..."):
                model  = load_model()
                result = predict(model, seq, alt, position)

            # ── Result display ─────────────────────────────────────────────
            st.divider()

            if result["prediction"] == "Pathogenic":
                st.error(
                    f"⚠️ **{result['prediction']}**  "
                    f"— Confidence: {result['confidence']}%"
                )
            else:
                st.success(
                    f"✅ **{result['prediction']}**  "
                    f"— Confidence: {result['confidence']}%"
                )

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Pathogenic Probability", f"{result['probability']:.3f}")
            col_b.metric("Prediction", result["prediction"])
            col_c.metric("Confidence", f"{result['confidence']}%")

            with st.expander("Show sequence context"):
                seq_display = result["sequence"]
                centre = FLANK
                st.code(
                    f"5'...{seq_display[:centre]}"
                    f"[{seq_display[centre]}]"
                    f"{seq_display[centre+1:]}...3'"
                )
                st.caption(
                    f"101bp flanking sequence (50bp either side). "
                    f"Bracketed base is the alternate allele at position {position}."
                )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
**Model:** 1D CNN trained on 341,825 ClinVar SNVs (GRCh38, germline, ⭐⭐+ review status)
· **Test AUC:** 0.73 · **Per-group equity gap:** −0.02

⚠️ *This tool is for educational and portfolio demonstration purposes only.
It is not a clinical diagnostic tool and should not inform medical decisions.*

[GitHub Repository](https://github.com/jorai-thomas/clinvar-classifier)
· Built by [Jorai Thomas](https://github.com/jorai-thomas)
""")