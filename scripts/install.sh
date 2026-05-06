#!/usr/bin/env bash
# Mac / Linux installer for clinic-codex
set -euo pipefail

REQUIRED_PY_MAJOR=3
REQUIRED_PY_MINOR=10
REQUIRED_NODE_MAJOR=18

command_exists() { command -v "$1" >/dev/null 2>&1; }

echo "Checking system prerequisites..."

if command_exists python3; then
  PY_VER_OUTPUT=$(python3 --version 2>&1)
  # python3 X.Y.Z
  PY_MAJOR=$(echo "$PY_VER_OUTPUT" | awk '{print $2}' | cut -d. -f1)
  PY_MINOR=$(echo "$PY_VER_OUTPUT" | awk '{print $2}' | cut -d. -f2)
  if [ "$PY_MAJOR" -lt "$REQUIRED_PY_MAJOR" ] || { [ "$PY_MAJOR" -eq "$REQUIRED_PY_MAJOR" ] && [ "$PY_MINOR" -lt "$REQUIRED_PY_MINOR" ]; }; then
    echo "ERROR: Python >= ${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR} is required. Found: $PY_VER_OUTPUT"
    exit 1
  fi
else
  echo "ERROR: python3 not found. Please install Python ${REQUIRED_PY_MAJOR}.${REQUIRED_PY_MINOR}+"
  exit 1
fi

if command_exists node; then
  NODE_VER_OUTPUT=$(node --version 2>&1)
  # vX.Y.Z
  NODE_MAJOR=$(echo "$NODE_VER_OUTPUT" | sed 's/^v//' | cut -d. -f1)
  if [ "$NODE_MAJOR" -lt "$REQUIRED_NODE_MAJOR" ]; then
    echo "ERROR: Node >= ${REQUIRED_NODE_MAJOR} is required. Found: $NODE_VER_OUTPUT"
    exit 1
  fi
else
  echo "ERROR: node not found. Please install Node ${REQUIRED_NODE_MAJOR}+"
  exit 1
fi

echo "Installing backend Python dependencies..."
if [ -f backend/requirements.txt ]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r backend/requirements.txt
else
  echo "WARNING: backend/requirements.txt not found — skipping pip install"
fi

if [ -d frontend ]; then
  echo "Installing frontend npm dependencies..."
  pushd frontend >/dev/null
  if command_exists npm; then
    npm install
  else
    echo "ERROR: npm not found. Install Node.js which includes npm."
    popd >/dev/null
    exit 1
  fi
  popd >/dev/null
else
  echo "WARNING: frontend/ directory not found — skipping npm install"
fi

echo "\n✓ Installation complete. Next steps:"
echo "  - Run the dev server: bash scripts/run-dev.sh"
echo "  - If you need to download model weights: bash scripts/download-weights.sh"
