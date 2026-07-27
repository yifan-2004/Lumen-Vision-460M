#!/usr/bin/env bash
# VisionPsyNano - OpenAI-compatible vLLM server (continuous batching).
#
# Note: the server expects pre-tiled images (one 512x512 tile per image_url item)
# and a prompt that already carries the global/tile position tokens with one
# image placeholder per tile. See infer.py for the reference client-side
# preprocessing.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=config.sh
source "$ROOT/scripts/config.sh"

CKPT="${CKPT:-$MODEL_ID}"
PORT="${PORT:-8900}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
LIMIT_MM_IMAGE="${LIMIT_MM_IMAGE:-128}"

die() { echo "ERROR: $*" >&2; exit 1; }

command -v vllm >/dev/null || die "vllm not found - run ./scripts/setup.sh and activate the venv"
# Local-dir checkpoints must exist; HF repo ids are resolved by vLLM at load time.
if [[ -d "$CKPT" ]]; then
  [[ -f "$CKPT/config.json" ]] || die "checkpoint dir has no config.json: $CKPT"
fi

echo "==== $MODEL_NAME - vLLM serve (:$PORT) ===="

exec vllm serve "$CKPT" \
    --served-model-name visionpsy-nano \
    --dtype float32 \
    --max-model-len "$MAX_MODEL_LEN" \
    --limit-mm-per-prompt "{\"image\":${LIMIT_MM_IMAGE}}" \
    --host 0.0.0.0 --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --enforce-eager
