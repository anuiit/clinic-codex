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

### Stockage des annotations

Les annotations sont sauvegardées directement dans le dossier du backend comme source unique de vérité.

- **Emplacement** : `backend/annotations/<analysis_id>/`
- **Fichiers créés** :
  - `image.png` : L'image originale de l'analyse.
  - `elements/el_<idx>.png` : Chaque recadrage d'élément annoté.
  - `metadata.json` : Contient les boîtes englobantes, les étiquettes et les horodatages.

Contrairement aux versions précédentes, le backend n'écrit plus directement dans `training_data/` et n'utilise plus de liens symboliques (symlinks).

### Option B: Standalone export

Use this when the backend is not running, or to export in bulk from the command line.

```bash
# Export a single analysis by ID
backend/.venv/bin/python3 scripts/export_annotations.py --analysis-id <id>

# Export all analyses from localStorage JSON dump
backend/.venv/bin/python3 scripts/export_annotations.py --all --input analyses.json
```

---

## Step 3 — Migration et Nettoyage

Si vous effectuez une mise à jour depuis une version plus ancienne utilisant des liens symboliques, vous devez nettoyer votre environnement :

1. **Vérifier les liens orphelins** :
   ```bash
   python scripts/migrate_annotation_symlinks.py --dry-run
   ```
2. **Appliquer le nettoyage** :
   ```bash
   python scripts/migrate_annotation_symlinks.py --apply
   ```

Cette étape supprime les anciens liens symboliques dans `training_data/Elements/` qui pointaient vers des fichiers déplacés ou supprimés.

---

## Step 4 — Retrain

Once crops are on disk under `backend/annotations/`:

```bash
bash scripts/retrain.sh
```

This runs the four pipeline steps in order:

| Step | Script | What it does |
|------|--------|--------------|
| 1/4 | `build_metadata.py` | Scans `annotations/` and writes `metadata.json` |
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
