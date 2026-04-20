#!/usr/bin/env bash
# Setup script for parallelism experiments.
# Clones torchtitan at the pinned commit and applies our patch.

set -e

TORCHTITAN_COMMIT="73a0e6979dd10b6b1904098eb3c8f62c18ab87ce"
TORCHTITAN_REPO="https://github.com/pytorch/torchtitan.git"

echo "== Setting up environment =="

# Create venv if missing
if [ ! -d venv ]; then
    echo "Creating virtualenv..."
    python3.11 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip

# Clone torchtitan at pinned commit
if [ ! -d torchtitan ]; then
    echo "Cloning torchtitan..."
    git clone "$TORCHTITAN_REPO"
fi

cd torchtitan
git fetch --all
git checkout "$TORCHTITAN_COMMIT"

# Check if our patch is already applied (idempotent re-runs)
if git diff --quiet HEAD -- torchtitan/train.py torchtitan/models/llama3/__init__.py; then
    echo "Applying patch..."
    git apply ../patches/csv_logging_and_1b_flavor.patch
else
    echo "Patch already applied (working tree dirty); skipping"
fi

# Install torchtitan's deps + torchtitan itself (editable)
pip install -r requirements.txt
pip install -e .

cd ..

# Install our analysis deps
pip install -r requirements.txt

echo ""
echo "== Setup complete =="
echo "Activate the venv with: source venv/bin/activate"
echo "Run experiments with:  python scripts/run_all.py"
