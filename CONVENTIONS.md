# Conventions

This file is the single source of truth for naming, structure, and shared
decisions in this repo. Update it when conventions change — don't let it drift.

---

## File structure

| Path | Purpose |
|---|---|
| `config.py` | All paths, hyperparameters, label mappings — import from here, never hardcode |
| `data/raw/` | Downloaded source files — treat as read-only, never modified by code |
| `data/processed/` | Filtered and encoded data — reproducible from raw via scripts |
| `notebooks/` | Exploratory work — numbered sequentially (`01_eda.ipynb`) |
| `src/` | All importable Python modules |
| `outputs/figures/` | All plots saved here — never displayed only, always saved |
| `outputs/predictions/` | Structured JSON outputs from model inference |
| `tests/` | Sanity checks and unit tests |

---

## Naming conventions

- **Files:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions and variables:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE` (defined in config.py)
- **Notebooks:** `NN_description.ipynb` — numbered to reflect order of execution

---

## Coding rules

- All paths via `config.py` — never hardcode a path anywhere else
- All paths constructed with `pathlib.Path` — no string concatenation with `/` or `\`
- All random operations seeded with `RANDOM_SEED` from config
- All plots saved to `FIGURES_DIR` before display
- All model outputs serialised to JSON in `PREDICTIONS_DIR`
- Always save best model state to `outputs/` at end of training — `torch.save(best_model_state, config.OUTPUTS_DIR / "model_name.pt")`

---

## Imports

Standard import order (enforced by convention, not linter for now):
1. Standard library
2. Third party (torch, pandas, numpy, biopython)
3. Local (`from src.data_utils import ...`)

---

## Git hygiene

- Commit messages: `type: short description` — e.g. `feat: add one-hot encoder`, `fix: handle missing alleles`, `docs: update README`
- One logical change per commit — don't bundle unrelated changes
- Never commit to main directly — use branches for new features (relaxed rule for solo work, but keep commits clean)
- Never commit data files — `data/` is git-ignored

---

## Cross-project notes

This repo is part of a dissertation arc:
- **clinvar-classifier** (this repo) — foundation: real genomic data, confounder auditing, reusable pipeline
- **research-notes** (private) — running journal of dissertation-relevant observations
- **deconfounding** (Year 3) — adversarial deconfounding across TCGA + 1000 Genomes

Anything observed here that is relevant to the Year 3 project should be logged
to research-notes simultaneously.

