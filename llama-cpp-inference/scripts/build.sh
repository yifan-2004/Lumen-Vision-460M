#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/config.sh"
LLAMA="$ROOT/llama.cpp-custom"
BACKEND="${BACKEND:-cuda}"
case "${BACKEND,,}" in
  cuda|gpu)
    BACKEND=cuda
    BUILD="${BUILD_DIR:-$LLAMA/build_cuda}"
    CUDA_ARCH="${CUDA_ARCH:-90}"
    CMAKE_EXTRA=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    echo "Building $MODEL_NAME → $BUILD (CUDA_ARCH=$CUDA_ARCH)"
    ;;
  cpu)
    BACKEND=cpu
    BUILD="${BUILD_DIR:-$LLAMA/build_cpu}"
    CMAKE_EXTRA=(-DGGML_CUDA=OFF)
    echo "Building $MODEL_NAME → $BUILD (CPU)"
    ;;
  *)
    echo "ERROR: BACKEND must be cuda or cpu, got: $BACKEND" >&2
    exit 1
    ;;
esac

rm -rf "$BUILD"
cmake -S "$LLAMA" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  "${CMAKE_EXTRA[@]}" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_APP=OFF
cmake --build "$BUILD" --target llama-mtmd-cli -j"$(nproc 2>/dev/null || echo 4)"
echo "Done: $BUILD/bin/llama-mtmd-cli"
