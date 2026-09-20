#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
marker="$tmp/should-not-exist"
printf 'MODEL_DIR=%s\n' "\$(touch $marker)" > "$tmp/literal.env"
(
  ENV_FILE="$tmp/literal.env"
  source "$ROOT/scripts/load-env.sh"
  [[ "$MODEL_DIR" == "\$(touch $marker)" ]]
)
[[ ! -e "$marker" ]] || { echo 'dotenv value executed unexpectedly' >&2; exit 1; }
for line in 'UNSUPPORTED_KEY=value' 'USE_REPLAYSSM=1' 'MTP_TOKENS=5'; do
  printf '%s\n' "$line" > "$tmp/unsupported.env"
  if (ENV_FILE="$tmp/unsupported.env"; source "$ROOT/scripts/load-env.sh") 2>/dev/null; then
    echo "unsupported dotenv key was accepted: ${line%%=*}" >&2; exit 1
  fi
done
printf 'PORT=8000\nPORT=8001\n' > "$tmp/duplicate.env"
if (ENV_FILE="$tmp/duplicate.env"; source "$ROOT/scripts/load-env.sh") 2>/dev/null; then
  echo 'duplicate dotenv key was accepted' >&2; exit 1
fi
mkdir -p "$tmp/model" "$tmp/draft" "$tmp/cache"
printf '{}\n' > "$tmp/model/config.json"
printf '{}\n' > "$tmp/draft/config.json"
# shellcheck disable=SC2030 # fixture assignments are intentionally subshell-local
validate_profile() (
  MODEL_DIR="$tmp/model" DRAFT_DIR="$tmp/draft" CACHE_DIR="$tmp/cache" GPU_DEVICES=0,2 ACCEPT_DFLASH2_RESEARCH_LICENSE=1
  source "$ROOT/scripts/defaults.sh"
  while (($#)); do export "${1?}"; shift; done
  source "$ROOT/scripts/validate-config.sh"
)
base_image='ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:v0.6.0@sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b'
validate_profile MAX_MODEL_LEN=524288
for boundary in 524289 1048576; do
  if validate_profile "MAX_MODEL_LEN=$boundary" 2>/dev/null; then
    echo "unpatched base image accepted unsafe context: $boundary" >&2; exit 1
  fi
done
fake_image_id='sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
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
export PATH="$tmp/fakebin:$PATH"
export RUNTIME_IMAGE="$fake_image_id"
validate_profile
validate_profile MAX_MODEL_LEN=1
validate_profile MAX_MODEL_LEN=1048576
validate_profile DFLASH_TOKENS=1
validate_profile DFLASH_TOKENS=5
validate_profile UPSTREAM_PORT=18001
[[ "$(<"$ROOT/config/example.env")" != *$'\nUPSTREAM_PORT='* ]]
for unsafe in \
  'MAX_MODEL_LEN=0' 'MAX_MODEL_LEN=1048577' 'MAX_MODEL_LEN=1.5' \
  'MAX_IMAGES_PER_PROMPT=15' 'MAX_IMAGES_PER_PROMPT=17' \
  'MAX_VIDEOS_PER_PROMPT=1' 'DFLASH_TOKENS=0' 'DFLASH_TOKENS=6' \
  'DFLASH_KV_CACHE_DTYPE=float16' 'KV_CACHE_DTYPE=nvfp4_ds_mla' \
  'ACCEPT_DFLASH2_RESEARCH_LICENSE=0' \
  'ENABLE_EXPERT_PARALLEL=0' 'ENABLE_PREFIX_CACHING=0' \
  'VLLM_DCP_TOPK_OWNER_MERGE=1' 'VLLM_B12X_DCP_TOPK_OWNER_EXCHANGE=1' \
  'DECODE_CONTEXT_PARALLEL_SIZE=1' 'GPU_DEVICES=0,0' \
  'UPSTREAM_PORT=0' 'UPSTREAM_PORT=65536' 'UPSTREAM_PORT=8000' \
  'RUNTIME_IMAGE=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx:latest'; do
  if validate_profile "$unsafe" 2>/dev/null; then
    echo "unsafe v0.6 profile value was accepted: $unsafe" >&2; exit 1
  fi
done
# shellcheck disable=SC2016
malicious='DFLASH_TOKENS=$(touch /tmp/dflash-config-must-not-execute)'
if validate_profile "$malicious" 2>/dev/null; then
  echo 'malicious DFlash value was accepted' >&2; exit 1
fi
[[ ! -e /tmp/dflash-config-must-not-execute ]] || { echo 'configuration executed syntax' >&2; exit 1; }
printf 'DRAFT_DIR=%s\nDFLASH_TOKENS=5\n' "$tmp/draft" > "$tmp/v06.env"
(
  ENV_FILE="$tmp/v06.env"
  source "$ROOT/scripts/load-env.sh"
  # shellcheck disable=SC2031 # values are loaded and asserted in this subshell
  [[ "$DRAFT_DIR" == "$tmp/draft" && "$DFLASH_TOKENS" == 5 ]]
)
echo 'strict dotenv and v0.6 configuration tests passed'
