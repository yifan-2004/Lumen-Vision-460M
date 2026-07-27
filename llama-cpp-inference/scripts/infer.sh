#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/config.sh"

BACKEND="${BACKEND:-cuda}"
case "${BACKEND,,}" in
  cuda|gpu) BACKEND=cuda; DEFAULT_CLI="./llama.cpp-custom/build_cuda/bin/llama-mtmd-cli"; DEFAULT_NGL=99 ;;
  cpu)      BACKEND=cpu;  DEFAULT_CLI="./llama.cpp-custom/build_cpu/bin/llama-mtmd-cli";  DEFAULT_NGL=0 ;;
  *)
    echo "ERROR: BACKEND must be cuda or cpu, got: $BACKEND" >&2
    exit 1
    ;;
esac

CLI="${CLI:-$DEFAULT_CLI}"
LM="${LM:-$MODEL_DIR/lm.gguf}"
MMPROJ="${MMPROJ:-$MODEL_DIR/mmproj.gguf}"
IMAGE="${IMAGE:-img.jpg}"
PROMPT="${PROMPT:-Please describe the image.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
NGL="${NGL:-$DEFAULT_NGL}"
THREADS="${THREADS:-8}"
CTX="${CTX:-8192}"
TEMP="${TEMP:-0}"

if [[ -z "${MTMD_NO_UPSCALE+x}" ]]; then
  if [[ "$IS_FLASH" == "1" ]]; then
    MTMD_NO_UPSCALE=1
  else
    MTMD_NO_UPSCALE=0
  fi
fi

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -x "$CLI" ]]    || die "llama-mtmd-cli not found — run BACKEND=$BACKEND ./scripts/build.sh first"
[[ -f "$LM" ]]     || die "LM weights missing: $LM (place GGUFs under $MODEL_DIR/)"
[[ -f "$MMPROJ" ]] || die "mmproj weights missing: $MMPROJ (place GGUFs under $MODEL_DIR/)"
[[ -f "$IMAGE" ]]  || die "image missing: $IMAGE"

export LD_LIBRARY_PATH="$(dirname "$CLI")"
if [[ "$MTMD_NO_UPSCALE" == "1" ]]; then
  export MTMD_NO_UPSCALE=1
else
  unset MTMD_NO_UPSCALE
fi

echo "==== $MODEL_NAME — llama.cpp infer ===="
echo "  CLI:              $CLI"
echo "  backend:          $BACKEND"
echo "  mode:             $MODEL_SLUG  (FLASH=$IS_FLASH  MTMD_NO_UPSCALE=${MTMD_NO_UPSCALE:-0})"
echo "  LM:               $LM"
echo "  mmproj:           $MMPROJ"
echo "  image:            $IMAGE"
echo "  prompt:           $PROMPT"
echo "  ngl:              $NGL   threads: $THREADS   ctx: $CTX   max_tokens: $MAX_TOKENS"
echo "================================="

STDERR_FILE=$(mktemp)
STDOUT_FILE=$(mktemp)
trap 'rm -f "$STDERR_FILE" "$STDOUT_FILE"' EXIT

set +e
"$CLI" \
  -m "$LM" \
  --mmproj "$MMPROJ" \
  --image "$IMAGE" \
  -p "$PROMPT" \
  -n "$MAX_TOKENS" \
  --temp "$TEMP" \
  -t "$THREADS" \
  -c "$CTX" \
  --no-warmup \
  -ngl "$NGL" \
  >"$STDOUT_FILE" 2>"$STDERR_FILE"
RC=$?
set -e

STDERR=$(cat "$STDERR_FILE")
STDOUT=$(cat "$STDOUT_FILE")

TEXT=$(printf '%s' "$STDOUT" | sed -E 's/\x1b\[[0-9;?]*[A-Za-z]//g')
TEXT=$(printf '%s' "$TEXT" | sed -e 's/\[end of text\].*$//' -e 's/<|endoftext|>.*$//' -e 's/<|im_end|>.*$//')
TEXT=$(printf '%s' "$TEXT" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')

echo
echo "---- model output ----"
echo "$TEXT"
echo "----------------------"
echo "exit=$RC"

if [[ $RC -ne 0 ]]; then
  echo "$STDERR" | tail -40 >&2
  die "llama-mtmd-cli failed (exit=$RC)"
fi

if echo "$STDERR" | grep -qE 'IM2COL failed|unsupported toolchain|CUDA error'; then
  echo "$STDERR" | tail -20 >&2
  die "CUDA backend error (check GPU arch / driver)"
fi

if echo "$TEXT" | grep -qiE '^\[?(error|timeout)\]?$'; then
  die "model returned an error sentinel"
fi

exit 0
