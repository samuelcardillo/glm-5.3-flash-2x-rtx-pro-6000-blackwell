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
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  [[ "$IMAGE" == local/glm53-runtime-fixes:* ]] && { echo "Derived runtime image is not built locally: $IMAGE" >&2; exit 2; }
  docker pull "$IMAGE"
fi
if [[ "$IMAGE" == local/glm53-runtime-fixes:* ]]; then
  _recipe_label=$(docker image inspect "$IMAGE" --format '{{ index .Config.Labels "io.github.glm53.runtime.recipe.sha256" }}')
  [[ "$_recipe_label" == "${RUNTIME_IMAGE##*:}" ]] || { echo 'Derived runtime recipe label mismatch' >&2; exit 2; }
  unset _recipe_label
fi
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" != true ]] || { echo 'Container already running' >&2; exit 1; }
  docker rm "$CONTAINER_NAME" >/dev/null
fi
MTP_SCHEDULE='[[1,1,5],[2,2,4],[3,3,3],[4,4,2],[5,8,1],[9,16,0]]'
SPEC_CONFIG=$(python3 -c 'import json,sys; print(json.dumps({"method":"mtp","num_speculative_tokens":int(sys.argv[1]),"num_speculative_tokens_per_batch_size":json.loads(sys.argv[2])},separators=(",",":")))' "$MTP_TOKENS" "$MTP_SCHEDULE")
CACHE_ARGS=(--no-enable-prefix-caching --mamba-cache-mode none); [[ "$ENABLE_PREFIX_CACHING" == 1 ]] && CACHE_ARGS=(--enable-prefix-caching --mamba-cache-mode align)
REPLAY_ARGS=(); [[ "$USE_REPLAYSSM" == 1 ]] && REPLAY_ARGS=(--use-replayssm --replayssm-buffer-len "$REPLAYSSM_BUFFER_LEN")
exec docker run --rm --name "$CONTAINER_NAME" --init \
  --gpus "device=${GPU_DEVICES}" --ipc=host --shm-size 32g --publish "${BIND_ADDRESS}:${PORT}:8001" \
  --env HF_HUB_OFFLINE=1 --env CUDA_DEVICE_ORDER=PCI_BUS_ID --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env NCCL_DEBUG="$NCCL_DEBUG" --env VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  --env GLM53_MIXED_PREFILL_CHUNK="$MIXED_PREFILL_CHUNK" \
  --env VLLM_ADAPTIVE_MTP="$ADAPTIVE_MTP" --env VLLM_ADAPTIVE_MTP_HISTORY=16 \
  --env VLLM_ADAPTIVE_MTP_MIN_DEPTH="$ADAPTIVE_MTP_MIN_DEPTH" --env VLLM_ADAPTIVE_MTP_DECISION_WINDOW=8 \
  --env VLLM_ADAPTIVE_MTP_PROBE_INTERVAL=32 --env VLLM_ADAPTIVE_MTP_PROBE_INTERVAL_MAX=256 \
  --env VLLM_ADAPTIVE_MTP_LOAD_THRESHOLDS=0.28,0.45,0.55,0.75,0.90 \
  --env VLLM_ENABLE_PCIE_ALLREDUCE=1 --env VLLM_PCIE_ALLREDUCE_BACKEND=b12x --env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=384KB \
  --env VLLM_B12X_DCP_A2A=1 --env VLLM_USE_B12X_SPARSE_INDEXER=1 --env VLLM_USE_B12X_KPOOL_INDEXER=1 \
  --env VLLM_DCP_GLOBAL_TOPK=1 --env VLLM_DCP_QUERY_SPLIT=0 --env VLLM_DCP_TOPK_OWNER_MERGE=1 \
  --env VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=1 --env VLLM_B12X_DCP_TOPK_MIN_ROWS=128 \
  --env VLLM_B12X_DCP_TOPK_MAX_ROWS="$MAX_NUM_BATCHED_TOKENS" --env KV_FP8_ROPE=1 --env VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 \
  --env VLLM_EXL3_TRELLIS_MIN_M=1 --env VLLM_EXL3_TRELLIS_MAX_M=32 --env VLLM_EXL3_TRELLIS_BLOCK_M=8 \
  --env VLLM_EXL3_PREFILL_TRELLIS=1 --env VLLM_EXL3_PREFILL_BLOCK_M=64 --env VLLM_EXL3_PREFILL_CAPACITY=1024 \
  --env B12X_EXL3_BF16_EPILOGUE=1 --env B12X_EXL3_BF16_GEMV=1 --env VLLM_B12X_GLM_H64_QUERY_PROJ=auto --env VLLM_USE_B12X_MHC=auto \
  --volume "$MODEL_DIR:/model:ro" --volume "$CACHE_DIR:/root/.cache" "$IMAGE" /model \
  --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port 8001 --tensor-parallel-size 2 \
  --decode-context-parallel-size 2 --dcp-comm-backend ag_rs --speculative-config "$SPEC_CONFIG" "${REPLAY_ARGS[@]}" \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}" "${CACHE_ARGS[@]}" \
  --attention-backend B12X_MLA_SPARSE --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" --max-cudagraph-capture-size 64 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" --no-enable-flashinfer-autotune --default-chat-template-kwargs '{"enable_thinking":false}' \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45
