#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
[[ -f "$ENV_FILE" ]] || { echo "Missing $ENV_FILE; copy config/example.env to .env" >&2; exit 2; }
# shellcheck source=scripts/load-env.sh
source "$ROOT/scripts/load-env.sh"
# shellcheck source=scripts/defaults.sh
source "$ROOT/scripts/defaults.sh"
# shellcheck source=scripts/validate-config.sh
source "$ROOT/scripts/validate-config.sh"
IMAGE="$RUNTIME_IMAGE"
mkdir -p "$CACHE_DIR"
DERIVED_TEMPLATE="$CACHE_DIR/chat_template.k3-thinking-control.jinja"
python3 "$ROOT/scripts/apply-thinking-template.py" "$DERIVED_TEMPLATE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker pull "$IMAGE"
fi
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != true ]] || { echo 'Container already running' >&2; exit 1; }
  docker rm "$CONTAINER_NAME" >/dev/null
fi
# shellcheck source=scripts/resolve-model-mount.sh
source "$ROOT/scripts/resolve-model-mount.sh"
resolve_model_mount "$MODEL_DIR" model
MODEL_MOUNT_SOURCE=$MOUNT_SOURCE; MODEL_MOUNT_TARGET=$MOUNT_TARGET; MODEL_CONTAINER_PATH=$CONTAINER_MODEL_PATH
resolve_model_mount "$DRAFT_DIR" draft
DRAFT_MOUNT_SOURCE=$MOUNT_SOURCE; DRAFT_MOUNT_TARGET=$MOUNT_TARGET; DRAFT_CONTAINER_PATH=$CONTAINER_MODEL_PATH
SPEC_CONFIG=$(python3 -c 'import json,sys; print(json.dumps({"method":"dflash","model":sys.argv[1],"num_speculative_tokens":int(sys.argv[2]),"kv_cache_dtype":sys.argv[3]},separators=(",",":")))' "$DRAFT_CONTAINER_PATH" "$DFLASH_TOKENS" "$DFLASH_KV_CACHE_DTYPE")
exec docker run --rm --name "$CONTAINER_NAME" --init \
  --gpus "\"device=${GPU_DEVICES}\"" --ipc=host --shm-size 32g --publish "${BIND_ADDRESS}:${PORT}:8001" \
  --env HF_HUB_OFFLINE=1 --env CUDA_DEVICE_ORDER=PCI_BUS_ID --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_DEBUG="$NCCL_DEBUG" --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env GLM53_STARTUP_WARMUP=1 --env GLM53_STARTUP_WARMUP_TIMEOUT_S=1800 \
  --env VLLM_ADAPTIVE_MTP=0 \
  --env VLLM_ENABLE_PCIE_ALLREDUCE=1 --env VLLM_PCIE_ALLREDUCE_BACKEND=b12x --env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=384KB \
  --env VLLM_B12X_PCIE_EAGER=0 --env VLLM_B12X_DCP_A2A=1 \
  --env VLLM_USE_B12X_SPARSE_INDEXER=1 --env VLLM_USE_B12X_KPOOL_INDEXER=1 \
  --env VLLM_DCP_GLOBAL_TOPK=1 --env VLLM_DCP_QUERY_SPLIT=0 --env VLLM_DCP_TOPK_OWNER_MERGE=1 \
  --env VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=1 --env VLLM_B12X_DCP_TOPK_MIN_ROWS=128 \
  --env VLLM_B12X_DCP_TOPK_MAX_ROWS="$MAX_NUM_BATCHED_TOKENS" --env KV_FP8_ROPE=0 --env VLLM_NVFP4_MLA_DYNAMIC_SCALE=0 \
  --env VLLM_EXL3_TRELLIS_MIN_M=1 --env VLLM_EXL3_TRELLIS_MAX_M=32 --env VLLM_EXL3_TRELLIS_BLOCK_M=8 \
  --env VLLM_EXL3_PREFILL_TRELLIS=1 --env VLLM_EXL3_PREFILL_BLOCK_M=64 --env VLLM_EXL3_PREFILL_CAPACITY=1024 \
  --env B12X_EXL3_BF16_EPILOGUE=1 --env B12X_EXL3_BF16_GEMV=1 \
  --env VLLM_B12X_GLM_H64_QUERY_PROJ=auto --env VLLM_USE_B12X_MHC=auto \
  --volume "$MODEL_MOUNT_SOURCE:$MODEL_MOUNT_TARGET:ro" --volume "$DRAFT_MOUNT_SOURCE:$DRAFT_MOUNT_TARGET:ro" --volume "$CACHE_DIR:/root/.cache" \
  --volume "$DERIVED_TEMPLATE:/config/chat_template.jinja:ro" \
  "$IMAGE" "$MODEL_CONTAINER_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port 8001 --tensor-parallel-size 2 \
  --enable-expert-parallel --decode-context-parallel-size "$DECODE_CONTEXT_PARALLEL_SIZE" --dcp-comm-backend ag_rs \
  --speculative-config "$SPEC_CONFIG" \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}" \
  --enable-prefix-caching --mamba-cache-mode align --attention-backend B12X_MLA_SPARSE \
  --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-cudagraph-capture-size 96 \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --no-enable-flashinfer-autotune --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 --chat-template /config/chat_template.jinja
