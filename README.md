# ClinVar Pathogenic Variant Classifier

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://clinvar-classifier-cfdyqhgeoh79yhepbrwafn.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-jorai--thomas-181717?logo=github)](https://github.com/jorai-thomas/clinvar-classifier)

A binary classifier distinguishing pathogenic from benign genetic variants
using the ClinVar dataset. Built as a technical foundation for adversarial
deconfounding research on population-stratified genomic data.

---

## Motivation

Clinical genomic AI tools are increasingly used to interpret genetic variants
and inform patient care. However, most training datasets — including ClinVar —
are heavily skewed toward European-ancestry populations and well-studied genes.
A model that performs well on its test set may fail silently when deployed on
patients from underrepresented populations, not because the model is wrong in
general, but because it never learned their biology.

This failure mode is not hypothetical. In 2021, the NHS retired the race-based
eGFR correction formula used in kidney function assessment after evidence showed
it systematically underestimated disease severity in Black patients — not due to
their actual biology, but due to a flawed assumption baked into the formula at
the point of calibration. The genomic equivalent is a model trained on
European-ancestry variants deployed on a patient of African or South Asian
descent. The model is miscalibrated for that person, and in a clinical setting
that is a patient safety issue.

This project builds the tooling and conceptual foundation to address that
problem. It is the first stage of a research arc culminating in adversarial
deconfounding across TCGA and 1000 Genomes populations.

---

## Key Finding

Adding gene identity as a feature tripled the performance gap between
well-studied and understudied genes — from 0.08 to 0.22 AUC points. A CNN
trained on flanking sequence context eliminated this gap entirely (-0.02),
demonstrating that sequence-based models generalise across the genome without
inheriting ClinVar's ascertainment bias.

| Model | Overall AUC | Top-50 Gene AUC | Other Gene AUC | Gap |
|---|---|---|---|---|
| Baseline MLP (allele only) | 0.595 | 0.655 | 0.577 | +0.08 |
| Enriched MLP (gene + chrom) | 0.669 | 0.827 | 0.613 | +0.22 |
| CNN (101bp flanking sequence) | 0.729 | 0.716 | 0.732 | -0.02 |
| CNN (301bp flanking sequence) | 0.725 | 0.717 | 0.726 | -0.01 |

---

## Project Objectives

1. Build a well-evaluated binary classifier (Pathogenic vs Benign) on
   high-confidence ClinVar variants
2. Audit the dataset for confounders — gene-level bias, ancestry
   ascertainment bias, review status bias — and document them explicitly
3. Demonstrate quantitatively that gene-identity encoding inherits
   database representation bias
4. Show that sequence-based encoding generalises equitably across
   gene representation groups
5. Produce a reproducible pipeline with structured outputs that feed
   directly into Year 3 deconfounding research

---

## Dataset

- Source: ClinVar `variant_summary.txt.gz` (June 2026)
- Filters: GRCh38, germline, ⭐⭐+ review status, unambiguous labels
- Raw variants: 8,985,214
- After filtering: 383,533 variants
- SNV subset: 341,825 (89.1%)
- Class ratio: 4.6:1 Benign:Pathogenic
- Split: 70/10/20 train/val/test (stratified)

---

## Design Decisions

### Why SNV-only?
Variable-length indels and deletions require distinct encoding strategies.
SNVs at 89% of the filtered dataset provide sufficient data for a meaningful
classifier. Crucially, SNVs are the most conserved variant class across
populations — any population stratification effects observed represent a
conservative lower bound on the true scale of the problem.

### Why top-50 genes + other bucket?
10,045 unique genes with a heavily long-tailed distribution (395 genes
cover only 50% of variants). Full one-hot encoding would create a sparse
10,045-dimensional vector. Top-50 + other preserves signal for well-studied
genes while explicitly representing the long tail as a single category —
itself informationally meaningful, as understudied genes disproportionately
represent variants from underrepresented populations.

Alternatives considered: embedding layers (more powerful but less
interpretable — obscures the confounder we are trying to measure),
hierarchical encoding (biologically principled but requires external
annotation databases).

### Why sequence context over gene identity?
Gene-identity encoding inherits ClinVar's ascertainment bias — the model
learns which genes are well-studied, not which mutations are biologically
damaging. Flanking sequence context encodes the actual DNA neighbourhood,
allowing the model to learn transferable biological rules (splice sites,
conserved motifs) that apply regardless of gene representation in clinical
databases.

### Why not a pretrained model (DNABERT, Nucleotide Transformer)?
Building a custom CNN from scratch makes the architecture transparent and
the contribution clear. Pretrained genomic foundation models are the
correct approach for Year 3, where fine-tuning and adversarial training
will be applied on top of a strong sequence representation.

---

## Stack

- Python 3.10+
- PyTorch — model training (CUDA on Windows, MPS on Apple Silicon, CPU fallback)
- pyfaidx — fast reference genome random access
- Biopython — sequence handling and NCBI Entrez API
- pandas / numpy — data processing
- scikit-learn — evaluation metrics
- Weights & Biases — experiment tracking
- Streamlit — live demo deployment
- huggingface_hub — model weight hosting
- pathlib — cross-platform path handling throughout

---

## Repository Structure
clinvar-classifier/
├── data/
│ ├── raw/ # Downloaded ClinVar files — git-ignored
│ └── processed/ # Filtered and encoded data — git-ignored
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory analysis and confounder audit
│ ├── 02_preprocessing.ipynb # SNV filter, one-hot encoding, splits
│ ├── 03_model.ipynb # Model training — MLP baseline, enriched MLP, CNN
│ └── 04_evaluation.ipynb # Full evaluation suite
├── src/
│ ├── data_utils.py # Loading, filtering, splitting
│ ├── features.py # Encoding functions
│ ├── model.py # PyTorch model definitions
│ ├── train.py # Training loop with W&B integration
│ └── evaluate.py # Metrics, plots, structured JSON outputs
├── outputs/
│ ├── figures/ # All evaluation plots
│ └── predictions/ # Structured JSON model outputs
├── app.py # Streamlit live demo
├── config.py # All paths and hyperparameters
└── CONVENTIONS.md # Naming and structure conventions

---

## Evaluation

Full evaluation suite in `notebooks/04_evaluation.ipynb`:
- ROC curves across all models
- Precision-recall curves (accounts for class imbalance)
- Calibration curves (note: all models require post-hoc calibration)
- Per-group AUC by gene representation — the core equity metric
- Confusion matrix at threshold 0.5

---

## Confounder Audit

A core output of this project is a documented audit of confounders present
in the ClinVar dataset:
- **Gene-level concentration** — BRCA1/2 alone = 3% of dataset. Top-50 genes
  account for a disproportionate share of variants, reflecting decades of
  research conducted predominantly on European-ancestry disease cohorts.
- **Ancestry ascertainment bias** — ClinVar submissions are heavily skewed
  toward European-ancestry variants. Variants common in African, South Asian,
  or other populations are systematically underrepresented.
- **Review status bias** — only 4.3% of raw variants pass the ⭐⭐+ quality
  filter. Low-star variants are disproportionately from understudied genes,
  which correlates with underrepresented populations.
- **Class imbalance** — 4.6:1 Benign:Pathogenic in SNV subset, reflecting
  clinical ascertainment bias toward disease-causing variants.

This audit directly motivates the adversarial deconfounding approach
in the follow-on research.

---

## Clinical Context — ACMG/AMP Criteria

The ACMG/AMP guidelines are the clinical standard for variant classification,
used by geneticists worldwide to assign pathogenicity. This project's findings
map directly onto two of the 28 criteria:

**PM2 — Absent from population databases**
A variant absent from or at very low frequency in population databases
(gnomAD, ExAC) is considered moderate evidence of pathogenicity. ClinVar's
European-ancestry skew means variants common in African or South Asian
populations may appear rare in reference databases — triggering PM2
incorrectly. A model trained on ClinVar inherits this miscalibration.

**BS1 — Allele frequency too high for disease**
A variant present at high frequency in population databases is considered
strong evidence of benignity. Frequency estimates derived from predominantly
European cohorts will be unreliable for other populations — a genuinely rare
pathogenic variant in an underrepresented group may appear common in a
European-skewed database, triggering BS1 incorrectly.

The per-group AUC gap documented in this project (Enriched MLP: +0.22
between well-studied and understudied genes) is a quantitative signature
of exactly this failure mode — the model is less reliable for variants in
genes whose population frequency landscape is poorly characterised.

Integrating gnomAD population-stratified allele frequencies directly into
the evaluation pipeline is the planned next step (see Project Plan).

---

## Part of a Research Arc

| Stage | Project | Status |
|---|---|---|
| 1 | ClinVar variant classifier (this repo) | ✅ Complete |
| 2 | Adversarial deconfounding across TCGA + 1000 Genomes | Planned — Year 3 dissertation |

---

## Reproducibility

**Clone and run notebooks:**
```bash
git clone https://github.com/jorai-thomas/clinvar-classifier.git
cd clinvar-classifier
pip install pandas numpy matplotlib seaborn jupyter ipykernel scikit-learn pyfaidx biopython torch wandb
```

Download required data files:
- ClinVar: `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`
- Reference genome: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz`

Save both to `data/raw/` then run notebooks in order (01 → 04).

**Run the live demo locally:**
```bash
pip install streamlit torch biopython numpy scikit-learn huggingface_hub
streamlit run app.py
```

Or visit the deployed demo: [clinvar-classifier.streamlit.app](https://clinvar-classifier-cfdyqhgeoh79yhepbrwafn.streamlit.app/)

---

## Author

**Jorai Thomas** — BSc Computer Science, University of Surrey
Researching population-aware genomic AI for clinical applications.
[GitHub](https://github.com/jorai-thomas/clinvar-classifier)