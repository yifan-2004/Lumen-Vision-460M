#!/usr/bin/env bash
# Create a virtualenv with vLLM + the VisionPsyNano plugin.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=config.sh
source "$ROOT/scripts/config.sh"

VENV="${VENV:-$ROOT/.venv}"
PYTHON="${PYTHON:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.22.0}"
# Default PyPI wheels target the newest CUDA toolkit. On machines with an
# older driver, point VLLM_PACKAGE at a matching build instead, e.g. the
# +cu129 wheel from vLLM's GitHub releases.
VLLM_PACKAGE="${VLLM_PACKAGE:-vllm==$VLLM_VERSION}"

echo "Setting up $MODEL_NAME vLLM runtime -> $VENV ($VLLM_PACKAGE)"
"$PYTHON" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --upgrade pip
pip install "$VLLM_PACKAGE"
pip install -e "$ROOT"
echo "Done. Activate with: source $VENV/bin/activate"
