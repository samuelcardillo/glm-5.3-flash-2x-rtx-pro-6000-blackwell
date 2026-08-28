#!/usr/bin/env bash
# Strict dotenv reader. This file is sourced; the operator-selected ENV_FILE is data only.
set -euo pipefail
_ALLOWED_KEYS=' MODEL_DIR CACHE_DIR GPU_DEVICES BIND_ADDRESS PORT CONTAINER_NAME SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS GPU_MEMORY_UTILIZATION KV_CACHE_DTYPE MAX_IMAGES_PER_PROMPT MAX_VIDEOS_PER_PROMPT ENABLE_PREFIX_CACHING MTP_TOKENS ADAPTIVE_MTP ADAPTIVE_MTP_MIN_DEPTH USE_REPLAYSSM REPLAYSSM_BUFFER_LEN NCCL_DEBUG '
declare -A _SEEN_ENV_KEYS=()
while IFS= read -r _line || [[ -n "$_line" ]]; do
  _line="${_line%$'\r'}"
  [[ -z "$_line" || "$_line" == '#'* ]] && continue
  [[ "$_line" == *'='* ]] || { echo "Invalid dotenv line (expected KEY=VALUE): $_line" >&2; return 2; }
  _key="${_line%%=*}"; _value="${_line#*=}"
  [[ "$_key" =~ ^[A-Z][A-Z0-9_]*$ ]] || { echo "Invalid dotenv key: $_key" >&2; return 2; }
  [[ "$_ALLOWED_KEYS" == *" $_key "* ]] || { echo "Unsupported dotenv key: $_key" >&2; return 2; }
  [[ -z "${_SEEN_ENV_KEYS[$_key]+x}" ]] || { echo "Duplicate dotenv key: $_key" >&2; return 2; }
  _SEEN_ENV_KEYS[$_key]=1
  printf -v "$_key" '%s' "$_value"
  export "${_key?}"
done < "$ENV_FILE"
unset _ALLOWED_KEYS _SEEN_ENV_KEYS _line _key _value
