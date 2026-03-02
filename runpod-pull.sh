#!/usr/bin/env bash
# Pull trained weights from a RunPod pod to local Mac.
#
# Usage:
#   ./runpod-pull.sh "ssh root@x.x.x.x -p 12345 -i ~/.ssh/runpod_ed25519"
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 \"ssh root@x.x.x.x -p 12345 -i ~/.ssh/runpod_ed25519\"" >&2
    exit 1
fi

SSH_CMD="$1"

# Parse host, port, and identity key from the SSH command string
HOST=$(echo "$SSH_CMD" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | head -1)
PORT=$(echo "$SSH_CMD" | grep -oE '\-p\s*[0-9]+' | grep -oE '[0-9]+' | head -1)
KEY=$(echo "$SSH_CMD" | grep -oE '\-i\s+[^ ]+' | sed 's/-i\s*//')

if [ -z "$HOST" ]; then
    echo "ERROR: could not parse user@host from: $SSH_CMD" >&2
    exit 1
fi
if [ -z "$PORT" ]; then
    PORT="22"
fi

SCP_OPTS="-P $PORT"
if [ -n "$KEY" ]; then
    SCP_OPTS="$SCP_OPTS -i $KEY"
fi

REMOTE_DIR="/workspace/TonicNet-PyTorch"
LOCAL_DIR="weights-runpod/$(date +%Y-%m-%d-%H%M%S)"

echo "=== Pulling weights from $HOST (port $PORT) ==="
mkdir -p "$LOCAL_DIR"

scp $SCP_OPTS \
    "${HOST}:${REMOTE_DIR}/tonicnet-best.pt" \
    "${HOST}:${REMOTE_DIR}/tonicnet-best.csv" \
    "$LOCAL_DIR/"

echo "=== Downloaded to $LOCAL_DIR/ ==="
ls -lh "$LOCAL_DIR/"
