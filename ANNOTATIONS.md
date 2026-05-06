# Annotations & Retraining Workflow

> How to correct model predictions and retrain Codex with your annotations.

## TL;DR

1. Open an analysis in the browser → adjust bounding boxes and class labels.
2. Click **"Envoyer pour entraînement"** (or run `scripts/export_annotations.py`).
3. Run `bash scripts/retrain.sh` to retrain the model on the new crops.

---

## Step 1 — Annotate in the browser

1. From the dashboard, open any analysis and click **"Aller à l'annotation"**.
2. In the annotation view:
   - **Select mode** (default): click a bounding box to select it, then change its class in the dropdown.
   - **Draw mode**: click the pen icon to draw new bounding boxes on the image.
   - **Delete**: click the trash icon on any element card to remove it.
3. Click **"Enregistrer les modifications"** to persist changes to localStorage.

---

## Step 2 — Send annotations to disk

Two options:
- **Live upload** (backend running): click "Envoyer pour entraînement"
- **Batch export** (offline): use `scripts/export_annotations.py`

### Option A: Live upload

With the Flask backend running (`bash scripts/run-dev.sh`):

1. Open the annotation view for an analysis.
2. Click **"Envoyer pour entraînement"** in the top toolbar.
3. A green toast confirms: `Envoyé : N éléments dans K classes`.

This calls `POST /save-annotation` on the backend, which:
- Decodes each element's crop from the image data URL.
- Saves crops to `backend/annotations/<analysis_id>/elements/`.
- Creates symlinks in `backend/training_data/Elements/<class_name>/`.

### Option B: Standalone export

Use this when the backend is not running, or to export in bulk from the command line.

```bash
# Export a single analysis by ID
backend/.venv/bin/python3 scripts/export_annotations.py --analysis-id <id>

# Export all analyses from localStorage JSON dump
backend/.venv/bin/python3 scripts/export_annotations.py --all --input analyses.json
```

The script reads from `localStorage` export files and writes the same layout as Option A.

---

## Step 3 — Retrain

Once crops are on disk under `backend/training_data/Elements/`:

```bash
bash scripts/retrain.sh
```

This runs the four pipeline steps in order:

| Step | Script | What it does |
|------|--------|--------------|
| 1/4 | `build_metadata.py` | Scans `training_data/` and writes `metadata.json` |
| 2/4 | `precompute_embeddings.py` | Extracts DINOv2 embeddings for all crops |
| 3/4 | `train.py` | Trains prototype classifiers |
| 4/4 | `export_model.py` | Writes `codex_model/weights/{prototypes.pt,projection.pt}` |

A lockfile at `backend/.retrain.lock` prevents concurrent runs. If a previous run crashed, delete the lockfile manually:

```bash
rm -f backend/.retrain.lock
```

> **Note**: The running Flask server does **not** hot-reload new weights. Restart the backend after retraining to pick up the new model.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `POST /save-annotation` returns 400 | Missing required field in payload | Check browser console for the error message |
| `POST /save-annotation` returns 413 | Image data URL too large (>50 MB) | Reduce image resolution before uploading |
| `retrain.sh` exits immediately with "already running" | Stale lockfile | `rm -f backend/.retrain.lock` |
| Crops look wrong (wrong region cropped) | Bbox in wrong format | Bbox must be `[x, y, width, height]` in image pixels |
| New weights not used after retraining | Flask server not restarted | Restart with `bash scripts/run-dev.sh` |
