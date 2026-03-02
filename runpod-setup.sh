#!/usr/bin/env bash
# RunPod GPU pod setup — paste into the RunPod web terminal.
# Clones repo, installs deps, verifies dataset & GPU, then prints the training command.
# Recommended pod: A40 ($0.20/hr spot) — model is <1M params, no need for more VRAM.
# SSH: ssh root@<host> -p <port> -i ~/.ssh/runpod_ed25519
set -euo pipefail

REPO="https://github.com/josmithiii/TonicNet-PyTorch.git"
BRANCH="xformer"
WORKDIR="/workspace/TonicNet-PyTorch"

echo "=== Cloning repo ==="
git clone "$REPO" "$WORKDIR"
cd "$WORKDIR"
git checkout "$BRANCH"

echo "=== Installing Python deps ==="
pip install music21 note-seq

echo "=== Verifying dataset ==="
if [ ! -f dataset_train.p ]; then
    echo "ERROR: dataset_train.p not found — aborting" >&2
    exit 1
fi
echo "dataset_train.p OK ($(du -h dataset_train.p | cut -f1))"

echo "=== Environment info ==="
nvidia-smi
echo ""
python3 --version
python3 -c "import torch; print(f'PyTorch {torch.__version__}  CUDA available: {torch.cuda.is_available()}  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Ready! Run: ==="
echo "  cd $WORKDIR && python train.py --overwrite --epochs 150"
