# CLAUDE.md — ClinVar Classifier

## Project state
Preprocessing complete. Ready to build PyTorch model.

## Completed notebooks
- `notebooks/01_eda.ipynb` — EDA, filtering, confounder audit
- `notebooks/02_preprocessing.ipynb` — SNV filter, one-hot encoding, splits

## Next notebook
`notebooks/03_model.ipynb` — PyTorch model, DataLoader, training loop, W&B

## Key files
- `config.py` — all paths and hyperparameters, edit here not in notebooks
- `data/processed/` — X_train/val/test.npy and y_train/val/test.npy ready to load
- `CONVENTIONS.md` — naming and structure rules

## Key decisions made
- SNV-only (89.1% of data) — variable length variants deferred
- One-hot encoding: 8-dimensional vector (ref 4 + alt 4)
- Stratified 70/10/20 split, 4.6:1 class ratio preserved
- Class imbalance handled via weighted loss function at training time
- Custom PyTorch architecture — no pretrained models at this stage

## Do not
- Hardcode any paths — use config.py
- Load full variant_summary.txt.gz again — use clinvar_filtered.tsv.gz
- Change split files — regenerating splits would break reproducibility