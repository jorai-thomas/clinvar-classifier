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

## Project objectives

1. Build a well-evaluated binary classifier (Pathogenic vs Benign) on
   high-confidence ClinVar variants
2. Audit the dataset for confounders — gene-level bias, ancestry
   ascertainment bias, review status bias — and document them explicitly
3. Produce a reproducible, well-documented pipeline with structured outputs
   that feed directly into the Year 3 deconfounding project

---

## Stack

- Python 3.10+
- PyTorch — model training
- Biopython — sequence handling
- pandas / numpy — data processing
- scikit-learn — evaluation metrics
- Weights & Biases — experiment tracking
- pathlib — cross-platform path handling throughout

---

## Repository structure
clinvar-classifier/

├── data/

│   ├── raw/          # Downloaded ClinVar files — git-ignored

│   └── processed/    # Filtered and encoded data — git-ignored

├── notebooks/        # Exploratory analysis — numbered sequentially

├── src/

│   ├── data_utils.py # Loading, filtering, splitting

│   ├── features.py   # One-hot encoding and feature engineering

│   ├── model.py      # PyTorch model definitions

│   ├── train.py      # Training loop

│   └── evaluate.py   # Metrics, plots, structured JSON outputs

├── outputs/

│   ├── figures/      # ROC curves, calibration plots, confusion matrices

│   └── predictions/  # Structured JSON model outputs

├── tests/            # Sanity checks and unit tests

├── config.py         # All paths, hyperparameters, label mappings

└── CONVENTIONS.md    # Naming and structure conventions

---

## Evaluation targets

Minimum: AUC, confusion matrix, calibration curve
Extended: precision-recall curve (given class imbalance), SHAP explainability

---

## Confounder Audit

A core output of this project is a documented audit of confounders present
in the ClinVar dataset:
- Gene-level concentration (BRCA1/2 alone = 3% of dataset)
- Ancestry ascertainment bias (European-ancestry enrichment)
- Class imbalance (4.6:1 Benign:Pathogenic in SNV subset)
- Review status bias (4.3% retention after ⭐⭐+ filter)

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
strong evidence of benignity. Again, frequency estimates derived from
predominantly European cohorts will be unreliable for other populations —
a genuinely rare pathogenic variant in an underrepresented group may appear
common in a European-skewed database, triggering BS1 incorrectly.

The per-group AUC gap documented in this project (Enriched MLP: +0.22
between well-studied and understudied genes) is a quantitative signature
of exactly this failure mode — the model is less reliable for variants in
genes whose population frequency landscape is poorly characterised.

Integrating gnomAD population-stratified allele frequencies directly into
the evaluation pipeline is the planned next step 

---

## Part of a research arc

This project is stage one of a two-stage research arc:

| Stage | Project | Status |
|---|---|---|
| 1 | ClinVar variant classifier (this repo) | In progress |
| 2 | Adversarial deconfounding across TCGA + 1000 Genomes | Planned — Year 3 dissertation |

## Reproducibility

```bash
git clone https://github.com/jorlyx/clinvar-classifier.git
cd clinvar-classifier
pip install -r requirements.txt
```

Data download instructions will be added after the data acquisition stage.

---

## Author

**Jorai** — BSc Computer Science, University of Surrey  
Researching population-aware genomic AI for clinical applications.  
[GitHub](https://github.com/jorai-thomas)