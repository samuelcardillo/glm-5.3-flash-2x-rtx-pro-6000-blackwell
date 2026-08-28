#!/usr/bin/env bash
# Validate public recipe configuration. Source after defaults are assigned.
set -euo pipefail
[[ "${MODEL_DIR:-}" == /* && -f "$MODEL_DIR/config.json" ]] || { echo 'MODEL_DIR must be absolute and contain config.json' >&2; return 2; }
[[ "${CACHE_DIR:-}" == /* ]] || { echo 'CACHE_DIR must be absolute' >&2; return 2; }
[[ "$GPU_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo 'GPU_DEVICES must look like 0,1 or 0,2' >&2; return 2; }
IFS=',' read -r _gpu_a _gpu_b <<< "$GPU_DEVICES"; [[ "$_gpu_a" != "$_gpu_b" ]] || { echo 'GPU devices must be distinct' >&2; return 2; }
[[ "$CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid CONTAINER_NAME' >&2; return 2; }
[[ "$SERVED_MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]*$ ]] || { echo 'Invalid SERVED_MODEL_NAME' >&2; return 2; }
for _v in PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS MAX_IMAGES_PER_PROMPT MAX_VIDEOS_PER_PROMPT MTP_TOKENS ADAPTIVE_MTP_MIN_DEPTH REPLAYSSM_BUFFER_LEN; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "$_v must be an integer" >&2; return 2; }
done
(( PORT>=1 && PORT<=65535 )) || { echo 'PORT must be 1..65535' >&2; return 2; }
(( MAX_MODEL_LEN>=1 && MAX_MODEL_LEN<=262144 )) || { echo 'MAX_MODEL_LEN must be 1..262144' >&2; return 2; }
(( MAX_NUM_BATCHED_TOKENS>=1 && MAX_NUM_BATCHED_TOKENS<=2048 )) || { echo 'MAX_NUM_BATCHED_TOKENS must be 1..2048' >&2; return 2; }
(( MAX_NUM_SEQS>=1 && MAX_NUM_SEQS<=16 )) || { echo 'MAX_NUM_SEQS must be 1..16' >&2; return 2; }
(( MAX_IMAGES_PER_PROMPT>=0 && MAX_IMAGES_PER_PROMPT<=16 )) || { echo 'MAX_IMAGES_PER_PROMPT must be 0..16' >&2; return 2; }
(( MAX_VIDEOS_PER_PROMPT==0 )) || { echo 'This qualified recipe requires MAX_VIDEOS_PER_PROMPT=0' >&2; return 2; }
(( MTP_TOKENS>=1 && MTP_TOKENS<=5 )) || { echo 'MTP_TOKENS must be 1..5' >&2; return 2; }
(( REPLAYSSM_BUFFER_LEN>=1 && REPLAYSSM_BUFFER_LEN<=10 )) || { echo 'REPLAYSSM_BUFFER_LEN must be 1..10' >&2; return 2; }
for _v in ENABLE_PREFIX_CACHING ADAPTIVE_MTP USE_REPLAYSSM; do
  [[ "${!_v}" == 0 || "${!_v}" == 1 ]] || { echo "$_v must be 0 or 1" >&2; return 2; }
done
(( ADAPTIVE_MTP_MIN_DEPTH>=1 && ADAPTIVE_MTP_MIN_DEPTH<=MTP_TOKENS )) || { echo 'ADAPTIVE_MTP_MIN_DEPTH must be within MTP depth' >&2; return 2; }
[[ "$KV_CACHE_DTYPE" == nvfp4_ds_mla ]] || { echo 'Qualified profile requires KV_CACHE_DTYPE=nvfp4_ds_mla' >&2; return 2; }
[[ "$NCCL_DEBUG" =~ ^(VERSION|WARN|INFO|TRACE|ABORT)$ ]] || { echo 'NCCL_DEBUG must be VERSION, WARN, INFO, TRACE, or ABORT' >&2; return 2; }
python3 - "$BIND_ADDRESS" "$GPU_MEMORY_UTILIZATION" <<'PY'
import ipaddress,sys
try: address=ipaddress.ip_address(sys.argv[1])
except ValueError as exc: raise SystemExit(f'BIND_ADDRESS must be a literal IPv4 address: {exc}')
if address.version != 4: raise SystemExit("BIND_ADDRESS must be IPv4; Docker's IPv6 publish syntax differs")
try: utilization=float(sys.argv[2])
except ValueError: raise SystemExit('GPU_MEMORY_UTILIZATION must be numeric')
if not 0 < utilization <= 0.950: raise SystemExit('GPU_MEMORY_UTILIZATION must be >0 and <=0.950')
PY
unset _gpu_a _gpu_b _v
