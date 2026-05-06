#!/usr/bin/env bash
# Download mobile_sam.pt into cache directory if missing
set -euo pipefail

URL="https://github.com/ChaoningZhang/MobileSAM/releases/download/v0.1.0/mobile_sam.pt"
CACHE_DIR="$HOME/.cache/mobile_sam"
TARGET="$CACHE_DIR/mobile_sam.pt"

if [ -f "$TARGET" ]; then
  echo "Weights already present at $TARGET"
  exit 0
fi

mkdir -p "$CACHE_DIR"

echo "Downloading mobile_sam.pt to $TARGET"

if command -v curl >/dev/null 2>&1; then
  curl -L --progress-bar -o "$TARGET" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TARGET" "$URL" --progress=bar:force
else
  echo "ERROR: neither curl nor wget found. Install one to download weights."
  exit 1
fi

echo "Download complete: $TARGET"
