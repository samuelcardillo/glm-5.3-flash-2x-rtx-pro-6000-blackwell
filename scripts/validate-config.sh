#!/usr/bin/env bash
# Validate the qualified tpurtell v0.6 / DFlash2 profile. Source after defaults.
set -euo pipefail
[[ "${MODEL_DIR:-}" == /* && -f "$MODEL_DIR/config.json" ]] || { echo 'MODEL_DIR must be absolute and contain config.json' >&2; return 2; }
[[ "${DRAFT_DIR:-}" == /* && -f "$DRAFT_DIR/config.json" ]] || { echo 'DRAFT_DIR must be absolute and contain config.json' >&2; return 2; }
[[ "$MODEL_DIR" != "$DRAFT_DIR" ]] || { echo 'MODEL_DIR and DRAFT_DIR must differ' >&2; return 2; }
[[ "${CACHE_DIR:-}" == /* ]] || { echo 'CACHE_DIR must be absolute' >&2; return 2; }
[[ "$GPU_DEVICES" =~ ^[0-9]+,[0-9]+$ ]] || { echo 'GPU_DEVICES must look like 0,1 or 0,2' >&2; return 2; }
IFS=',' read -r _gpu_a _gpu_b <<< "$GPU_DEVICES"
[[ "$_gpu_a" != "$_gpu_b" ]] || { echo 'GPU devices must be distinct' >&2; return 2; }
[[ "$CONTAINER_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo 'Invalid CONTAINER_NAME' >&2; return 2; }
[[ "$SERVED_MODEL_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._:/-]*$ ]] || { echo 'Invalid SERVED_MODEL_NAME' >&2; return 2; }
_base_runtime_image='ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0@sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b'
[[ "$RUNTIME_IMAGE" == "$_base_runtime_image" ]] || { echo 'RUNTIME_IMAGE must be the qualified immutable v0.6.0 digest' >&2; return 2; }
for _v in PORT MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS MAX_IMAGES_PER_PROMPT MAX_VIDEOS_PER_PROMPT DFLASH_TOKENS DECODE_CONTEXT_PARALLEL_SIZE; do
  [[ "${!_v}" =~ ^[0-9]+$ ]] || { echo "$_v must be an integer" >&2; return 2; }
done
(( PORT>=1 && PORT<=65535 )) || { echo 'PORT must be 1..65535' >&2; return 2; }
(( MAX_MODEL_LEN>=1 && MAX_MODEL_LEN<=1048576 )) || { echo 'MAX_MODEL_LEN must be 1..1048576' >&2; return 2; }
(( MAX_NUM_BATCHED_TOKENS>=1 && MAX_NUM_BATCHED_TOKENS<=2048 )) || { echo 'MAX_NUM_BATCHED_TOKENS must be 1..2048' >&2; return 2; }
(( MAX_NUM_SEQS>=1 && MAX_NUM_SEQS<=16 )) || { echo 'MAX_NUM_SEQS must be 1..16' >&2; return 2; }
(( MAX_IMAGES_PER_PROMPT==16 )) || { echo 'Qualified profile requires MAX_IMAGES_PER_PROMPT=16' >&2; return 2; }
(( MAX_VIDEOS_PER_PROMPT==0 )) || { echo 'Qualified profile requires MAX_VIDEOS_PER_PROMPT=0' >&2; return 2; }
(( DFLASH_TOKENS>=1 && DFLASH_TOKENS<=5 )) || { echo 'DFLASH_TOKENS must be 1..5' >&2; return 2; }
(( DECODE_CONTEXT_PARALLEL_SIZE==2 )) || { echo 'Qualified profile requires DECODE_CONTEXT_PARALLEL_SIZE=2' >&2; return 2; }
for _v in ENABLE_PREFIX_CACHING ENABLE_EXPERT_PARALLEL; do
  [[ "${!_v}" == 0 || "${!_v}" == 1 ]] || { echo "$_v must be 0 or 1" >&2; return 2; }
done
[[ "$ENABLE_PREFIX_CACHING" == 1 ]] || { echo 'Qualified profile requires prefix caching' >&2; return 2; }
[[ "$ENABLE_EXPERT_PARALLEL" == 1 ]] || { echo 'Qualified profile requires expert parallelism' >&2; return 2; }
[[ "$KV_CACHE_DTYPE" == fp8_ds_mla ]] || { echo 'Qualified profile requires KV_CACHE_DTYPE=fp8_ds_mla' >&2; return 2; }
[[ "$DFLASH_KV_CACHE_DTYPE" == bfloat16 ]] || { echo 'DFLASH_KV_CACHE_DTYPE must be bfloat16' >&2; return 2; }
[[ "$ACCEPT_DFLASH2_RESEARCH_LICENSE" == 1 ]] || { echo 'Set ACCEPT_DFLASH2_RESEARCH_LICENSE=1 after reviewing the DFlash2 CC BY-NC-ND 4.0 research/evaluation terms; commercial use requires separate permission' >&2; return 2; }
[[ "$NCCL_DEBUG" =~ ^(VERSION|WARN|INFO|TRACE|ABORT)$ ]] || { echo 'NCCL_DEBUG must be VERSION, WARN, INFO, TRACE, or ABORT' >&2; return 2; }
python3 - "$BIND_ADDRESS" "$GPU_MEMORY_UTILIZATION" <<'PY'
import ipaddress, sys
try:
    address = ipaddress.ip_address(sys.argv[1])
except ValueError as exc:
    raise SystemExit(f'BIND_ADDRESS must be a literal IPv4 address: {exc}')
if address.version != 4:
    raise SystemExit("BIND_ADDRESS must be IPv4; Docker's IPv6 publish syntax differs")
try:
    utilization = float(sys.argv[2])
except ValueError:
    raise SystemExit('GPU_MEMORY_UTILIZATION must be numeric')
if not 0 < utilization <= 0.950:
    raise SystemExit('GPU_MEMORY_UTILIZATION must be >0 and <=0.950')
PY
unset _gpu_a _gpu_b _v _base_runtime_image
