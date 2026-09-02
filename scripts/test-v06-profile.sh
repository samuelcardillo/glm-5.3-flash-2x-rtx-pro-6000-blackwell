#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/model" "$tmp/draft" "$tmp/cache"
printf '{}\n' > "$tmp/model/config.json"
printf '{}\n' > "$tmp/model/quantization_config.json"
printf '{}\n' > "$tmp/draft/config.json"
printf '' > "$tmp/draft/model.safetensors"
(
  MODEL_DIR="$tmp/model"
  DRAFT_DIR="$tmp/draft"
  CACHE_DIR="$tmp/cache"
  GPU_DEVICES=0,2
  source "$ROOT/scripts/defaults.sh"
  [[ "$MAX_MODEL_LEN" == 1048576 ]]
  [[ "$MAX_NUM_BATCHED_TOKENS" == 2048 ]]
  [[ "$MAX_NUM_SEQS" == 16 ]]
  [[ "$MAX_IMAGES_PER_PROMPT" == 16 ]]
  [[ "$MAX_VIDEOS_PER_PROMPT" == 0 ]]
  [[ "$KV_CACHE_DTYPE" == fp8_ds_mla ]]
  [[ "$DFLASH_TOKENS" == 5 ]]
  [[ "$DFLASH_KV_CACHE_DTYPE" == bfloat16 ]]
  [[ "$ACCEPT_DFLASH2_RESEARCH_LICENSE" == 0 ]]
  [[ "$ENABLE_PREFIX_CACHING" == 1 ]]
  [[ "$ENABLE_EXPERT_PARALLEL" == 1 ]]
  [[ "$DECODE_CONTEXT_PARALLEL_SIZE" == 2 ]]
  [[ "$RUNTIME_IMAGE" == 'ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0@sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b' ]]
  ACCEPT_DFLASH2_RESEARCH_LICENSE=1
  source "$ROOT/scripts/validate-config.sh"
)
download=$(<"$ROOT/scripts/download-model.sh")
[[ "$download" == *"ACCEPT_DFLASH2_RESEARCH_LICENSE"* ]]
[[ "$download" == *"before downloading"* ]]
serve=$(<"$ROOT/scripts/serve.sh")
[[ "$serve" == *'--default-chat-template-kwargs '* ]]
[[ "$serve" == *'{"enable_thinking":false}'* ]]
[[ "$serve" == *'apply-thinking-template.py'* ]]
[[ "$serve" == *'--chat-template /config/chat_template.jinja'* ]]
# shellcheck disable=SC2016 # literal source-code contracts, not expansions
[[ "$serve" == *'--publish "${BIND_ADDRESS}:${PORT}:8001"'* ]]
# shellcheck disable=SC2016 # literal source-code contracts, not expansions
[[ "$serve" == *'--gpus "\"device=${GPU_DEVICES}\""'* ]]
[[ "$serve" == *'--speculative-config'* ]]
[[ "$serve" == *'--enable-expert-parallel'* ]]
[[ "$serve" == *'--decode-context-parallel-size'* ]]
[[ "$serve" == *'--limit-mm-per-prompt'* ]]
[[ "$serve" == *'--max-cudagraph-capture-size 96'* ]]
[[ "$serve" != *'--use-replayssm'* ]]
echo 'v0.6 profile contract tests passed'
