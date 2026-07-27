#!/usr/bin/env bash
FLASH="${FLASH:-0}"
case "${FLASH,,}" in
  1|true|yes|flash) IS_FLASH=1 ;;
  0|false|no|nano|"") IS_FLASH=0 ;;
  *)
    echo "ERROR: FLASH must be 0/1 (or nano/flash), got: $FLASH" >&2
    exit 1
    ;;
esac

if [[ "$IS_FLASH" == "1" ]]; then
  MODEL_NAME="VisionPsyNano Flash"
  MODEL_SLUG="visionpsy-nano-flash"
  MODEL_DIR="${MODEL_DIR:-models/flash}"
else
  MODEL_NAME="VisionPsyNano"
  MODEL_SLUG="visionpsy-nano"
  MODEL_DIR="${MODEL_DIR:-models/nano}"
fi
