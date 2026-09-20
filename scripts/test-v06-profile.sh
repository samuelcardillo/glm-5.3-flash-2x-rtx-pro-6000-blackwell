#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/model" "$tmp/draft" "$tmp/cache"
printf '{}\n' > "$tmp/model/config.json"
printf '{}\n' > "$tmp/model/quantization_config.json"
printf '{}\n' > "$tmp/draft/config.json"
printf '' > "$tmp/draft/model.safetensors"
mkdir -p "$tmp/fakebin"
cat > "$tmp/fakebin/docker" <<'SH'
#!/usr/bin/env bash
case "$*" in
  *'{{.Id}}'*) printf '%s\n' 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' ;;
  *'org.nous.glm53.base-image-id'*) printf '%s\n' 'sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b' ;;
  *'org.nous.glm53.overlay-recipe-sha256'*) printf '%s\n' 'a9edc75da46621a05361ef42dd4ebfe7681eeb65566a1c2ae207757e3896101b' ;;
  *'org.nous.glm53.overlay'*) printf '%s\n' 'dflash-dcp-block-table' ;;
  *) exit 2 ;;
esac
SH
chmod 0755 "$tmp/fakebin/docker"
(
  PATH="$tmp/fakebin:$PATH"
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
  [[ "$VLLM_DCP_TOPK_OWNER_MERGE" == 0 ]]
  [[ "$VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE" == 0 ]]
  [[ "$UPSTREAM_PORT" == 18001 ]]
  [[ "$RUNTIME_IMAGE" == 'ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0@sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b' ]]
  RUNTIME_IMAGE='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
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
[[ "$serve" == *'--publish "127.0.0.1:${UPSTREAM_PORT}:8001"'* ]]
[[ "$serve" == *'tool-loop-guard.py'* ]]
[[ "$serve" == *'--upstream "http://127.0.0.1:${UPSTREAM_PORT}"'* ]]
[[ "$serve" == *'--cidfile "$CID_FILE"'* ]]
[[ "$serve" == *'--label "${MANAGED_LABEL_KEY}=${MANAGED_LABEL_VALUE}"'* ]]
# shellcheck disable=SC2016 # literal source-code contracts, not expansions
[[ "$serve" == *'--gpus "\"device=${GPU_DEVICES}\""'* ]]
[[ "$serve" == *'--speculative-config'* ]]
[[ "$serve" == *'--env VLLM_DCP_TOPK_OWNER_MERGE="$VLLM_DCP_TOPK_OWNER_MERGE"'* ]]
[[ "$serve" == *'--env VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE="$VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE"'* ]]
[[ "$serve" != *'--env VLLM_DCP_TOPK_OWNER_MERGE=1'* ]]
[[ "$serve" != *'--env VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=1'* ]]
[[ "$serve" == *'--enable-expert-parallel'* ]]
[[ "$serve" == *'--decode-context-parallel-size'* ]]
[[ "$serve" == *'--limit-mm-per-prompt'* ]]
[[ "$serve" == *'--max-cudagraph-capture-size 96'* ]]
[[ "$serve" != *'--use-replayssm'* ]]
unit=$(<"$ROOT/systemd/glm53-2x-rtxpro6000.service.in")
[[ "$unit" == *'ExecStop=/bin/kill -s INT $MAINPID'* ]]
[[ "$unit" == *'TimeoutStartSec=4500'* ]]
[[ "$unit" == *'TimeoutStopSec=240'* ]]
[[ "$unit" == *'--timeout 4200'* ]]
echo 'v0.6 profile contract tests passed'
