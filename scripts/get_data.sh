#!/usr/bin/env bash
# Download SpectralWaste and print the SW_ROOT line.
#
# Usage (from the repo root, or from a Colab cell):
#   bash scripts/get_data.sh [destination]
#
# destination defaults to ./data. The zip is about 21 GB and the unpacked
# folder is about 24 GB. Re-run the same command to resume a partial download.
set -euo pipefail

URL="https://zenodo.org/records/10880544/files/spectralwaste_segmentation.zip?download=1"
EXPECTED_MD5="ae2c268bb385345f12e55b4389d2d3b9"
DEST="${1:-./data}"

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
ZIP="$DEST/spectralwaste_segmentation.zip"

file_md5() {
  local f="$1"
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$f" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$f"
  else
    python3 - "$f" <<'PY'
import hashlib, sys
h = hashlib.md5()
with open(sys.argv[1], "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        h.update(block)
print(h.hexdigest())
PY
  fi
}

echo "Downloading SpectralWaste (~21 GB zip, ~24 GB unpacked) to:"
echo "  $ZIP"
echo "Re-run this script to resume if the download stops."

# Exit 33 means the server refused a range because the file is already complete.
status=0
curl -L --fail --retry 5 --retry-delay 2 --continue-at - -o "$ZIP" "$URL" || status=$?
if [[ "$status" -ne 0 && "$status" -ne 33 ]]; then
  echo "Download failed (curl exit $status)." >&2
  exit "$status"
fi

echo "Checking md5 (this reads the whole zip)..."
actual="$(file_md5 "$ZIP")"
if [[ "$actual" != "$EXPECTED_MD5" ]]; then
  echo "md5 mismatch: got $actual, expected $EXPECTED_MD5" >&2
  echo "Delete $ZIP and run this script again." >&2
  exit 1
fi

echo "Unzipping into $DEST"
unzip -n -q "$ZIP" -d "$DEST"

ROOT="$DEST/spectralwaste_segmentation"
if [[ ! -d "$ROOT/hyper" || ! -d "$ROOT/labels_hyper_lt" ]]; then
  echo "Unzip finished, but $ROOT is missing hyper/ or labels_hyper_lt/." >&2
  exit 1
fi

echo
echo "Done. In a shell:"
echo "export SW_ROOT=$ROOT"
echo
echo "In Python or Colab:"
echo "import os"
echo "os.environ[\"SW_ROOT\"] = \"$ROOT\""
