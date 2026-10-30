#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 CONFIG [GPU_IDS] [PYTHON]" >&2
  echo "example: $0 configs/qwen35_em_tool_ce_chainclean.json 0,1" >&2
  exit 2
fi

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$(readlink -f "$1")
GPU_IDS=${2:-0,1}
PYTHON=${3:-python}

if [ ! -f "$CONFIG" ]; then
  echo "config not found: $CONFIG" >&2
  exit 1
fi

WORLD_SIZE=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_world_size"])' "$CONFIG"
)
GPU_COUNT=$(awk -F, '{print NF}' <<<"$GPU_IDS")
if [ "$GPU_COUNT" -ne "$WORLD_SIZE" ]; then
  echo "GPU_IDS contains $GPU_COUNT devices but config expects $WORLD_SIZE ranks" >&2
  exit 1
fi

if nvidia-smi -i "$GPU_IDS" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "one or more requested GPUs already have compute processes: $GPU_IDS" >&2
  exit 1
fi

RUN_NAME=$(
  "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_name"])' "$CONFIG"
)
mkdir -p "$ROOT/logs"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
exec "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$WORLD_SIZE" \
  src/train_em_tool_ce.py "$CONFIG" \
  2>&1 | tee -a "logs/${RUN_NAME}.log"
