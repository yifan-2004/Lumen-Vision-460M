#!/usr/bin/env bash
# VisionPsyNano - one-shot image+prompt inference via vLLM.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=config.sh
source "$ROOT/scripts/config.sh"

CKPT="${CKPT:-$MODEL_ID}"
IMAGE="${IMAGE:-img.jpg}"
PROMPT="${PROMPT:-Please describe the image.}"
MAX_TOKENS="${MAX_TOKENS:-256}"

die() { echo "ERROR: $*" >&2; exit 1; }

python -c "import vllm" 2>/dev/null || die "vllm not importable - run ./scripts/setup.sh and activate the venv"
# Local-dir checkpoints must exist; HF repo ids are resolved by vLLM at load time.
if [[ -d "$CKPT" ]]; then
  [[ -f "$CKPT/config.json" ]] || die "checkpoint dir has no config.json: $CKPT"
fi
[[ -f "$IMAGE" ]] || die "image missing: $IMAGE"

echo "==== $MODEL_NAME - vLLM infer ===="
echo "  ckpt:       $CKPT"
echo "  image:      $IMAGE"
echo "  prompt:     $PROMPT"
echo "  max_tokens: $MAX_TOKENS"
echo "================================="

python infer.py \
  --ckpt "$CKPT" \
  --image "$IMAGE" \
  --prompt "$PROMPT" \
  --max-tokens "$MAX_TOKENS"
