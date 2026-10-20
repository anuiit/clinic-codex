# scripts/

## Maintained tools (this directory)
Installation and run (`install.*`, `run-dev.*`, `retrain.*`, `download-weights.sh`, `check-*`),
model lifecycle (`promote_model.py`, `build_*`, `freeze_*`, `export_*`, `import_*`, `evaluate_*`,
`train_*`), their tests (`test_*.py`) and the `autoresearch_*`, `tlachia_*` and `lot_a/` research code
that `backend/tests` imports by path.

## experiments/
One-shot research scripts kept for reproducibility of the 2026-10 experiments: `research_*`,
`report_*`, `summarize_*`, `plot_*` and the two tests `test_research_recognition_*.py`.
Run them from the repository root, e.g. `python -m pytest scripts/experiments` or
`python scripts/experiments/research_<name>.py`. Many `report_*` and `plot_*` scripts execute at
import time and write to `reports/`; do not import them casually.
They import maintained tools with `from scripts.<tool> import ...` and each other with
`from scripts.experiments.<module> import ...`.