#!/usr/bin/env bash
# Pull trained weights from a RunPod pod to local Mac.
#
# Usage:
#   ./runpod-pull.sh "ssh root@x.x.x.x -p 12345"
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 \"ssh root@x.x.x.x -p 12345\"" >&2
    exit 1
fi

SSH_CMD="$1"

# Parse host and port from "ssh root@host -p port" or "ssh -p port root@host"
HOST=$(echo "$SSH_CMD" | grep -oE '[a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+' | head -1)
PORT=$(echo "$SSH_CMD" | grep -oE '\-p\s*[0-9]+' | grep -oE '[0-9]+' | head -1)

if [ -z "$HOST" ]; then
    echo "ERROR: could not parse user@host from: $SSH_CMD" >&2
    exit 1
fi
if [ -z "$PORT" ]; then
    PORT="22"
fi

REMOTE_DIR="/workspace/TonicNet-PyTorch"
LOCAL_DIR="weights-runpod/$(date +%Y-%m-%d-%H%M%S)"

echo "=== Pulling weights from $HOST (port $PORT) ==="
mkdir -p "$LOCAL_DIR"

scp -P "$PORT" \
    "${HOST}:${REMOTE_DIR}/tonicnet-best.pt" \
    "${HOST}:${REMOTE_DIR}/tonicnet-best.csv" \
    "$LOCAL_DIR/"

echo "=== Downloaded to $LOCAL_DIR/ ==="
ls -lh "$LOCAL_DIR/"
