#!/usr/bin/env bash
set -euo pipefail
TARGET_REPO='wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1'
TARGET_REVISION='319d66a8b53092b491f698440ecea781e4ddd4e4'
DRAFT_REPO='incoai/GLM-5.3-Flash-DFlash2'
DRAFT_REVISION='dc77ff1c99eeb2df044ee3d4f0094eb033fee410'
[[ "${ACCEPT_DFLASH2_RESEARCH_LICENSE:-0}" == 1 ]] || { echo 'Set ACCEPT_DFLASH2_RESEARCH_LICENSE=1 after reviewing the DFlash2 CC BY-NC-ND 4.0 research/evaluation terms before downloading; commercial use requires separate permission.' >&2; exit 2; }
TARGET_DESTINATION="${1:-${MODEL_DIR:-}}"
DRAFT_DESTINATION="${2:-${DRAFT_DIR:-}}"
[[ -n "$TARGET_DESTINATION" && "$TARGET_DESTINATION" == /* ]] || { echo 'Pass an absolute target destination or set MODEL_DIR.' >&2; exit 2; }
[[ -n "$DRAFT_DESTINATION" && "$DRAFT_DESTINATION" == /* ]] || { echo 'Pass an absolute draft destination or set DRAFT_DIR.' >&2; exit 2; }
[[ "$TARGET_DESTINATION" != "$DRAFT_DESTINATION" ]] || { echo 'Target and draft destinations must differ.' >&2; exit 2; }
command -v hf >/dev/null || { echo 'Missing Hugging Face hf CLI.' >&2; exit 1; }
hf download "$TARGET_REPO" --revision "$TARGET_REVISION" --local-dir "$TARGET_DESTINATION"
hf download "$DRAFT_REPO" --revision "$DRAFT_REVISION" --local-dir "$DRAFT_DESTINATION"
printf '%s\n' "${TARGET_REPO}@${TARGET_REVISION}" > "$TARGET_DESTINATION/RECIPE_PIN.txt"
printf '%s\n' "${DRAFT_REPO}@${DRAFT_REVISION}" > "$DRAFT_DESTINATION/RECIPE_PIN.txt"
printf 'Downloaded target %s and DFlash2 draft %s at immutable revisions.\n' "$TARGET_REPO" "$DRAFT_REPO"
