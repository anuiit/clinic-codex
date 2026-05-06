#!/usr/bin/env bash
# retrain.sh — run the full retraining pipeline with a lockfile guard
# Usage: bash scripts/retrain.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCKFILE="$REPO_ROOT/backend/.retrain.lock"
PYTHON="$REPO_ROOT/backend/.venv/bin/python3"
PIPELINE="$REPO_ROOT/backend/codex_pipeline/scripts"

# --- Lockfile guard ---
if [ -f "$LOCKFILE" ]; then
  OLD_PID=$(cat "$LOCKFILE")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "ERROR: retrain already running (PID $OLD_PID). Aborting." >&2
    exit 1
  else
    echo "WARNING: stale lockfile (PID $OLD_PID not running). Removing." >&2
    rm -f "$LOCKFILE"
  fi
fi

echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT INT TERM

echo "=== [1/4] build_metadata ==="
"$PYTHON" "$PIPELINE/build_metadata.py" || exit 1

echo "=== [2/4] precompute_embeddings ==="
"$PYTHON" "$PIPELINE/precompute_embeddings.py" || exit 1

echo "=== [3/4] train ==="
"$PYTHON" "$PIPELINE/train.py" || exit 1

echo "=== [4/4] export_model ==="
"$PYTHON" "$PIPELINE/export_model.py" || exit 1

echo "=== Retraining complete ==="
