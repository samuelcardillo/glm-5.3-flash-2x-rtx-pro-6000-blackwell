#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
marker="$tmp/should-not-exist"
printf 'MODEL_DIR=%s\n' "\$(touch $marker)" > "$tmp/literal.env"
(
  ENV_FILE="$tmp/literal.env"
  # shellcheck source=scripts/load-env.sh
  source "$ROOT/scripts/load-env.sh"
  [[ "$MODEL_DIR" == "\$(touch $marker)" ]]
)
[[ ! -e "$marker" ]] || { echo 'dotenv value executed unexpectedly' >&2; exit 1; }
printf 'UNSUPPORTED_KEY=value\n' > "$tmp/unsupported.env"
if (ENV_FILE="$tmp/unsupported.env"; source "$ROOT/scripts/load-env.sh") 2>/dev/null; then
  echo 'unsupported dotenv key was accepted' >&2; exit 1
fi
printf 'PORT=8000\nPORT=8001\n' > "$tmp/duplicate.env"
if (ENV_FILE="$tmp/duplicate.env"; source "$ROOT/scripts/load-env.sh") 2>/dev/null; then
  echo 'duplicate dotenv key was accepted' >&2; exit 1
fi
mkdir -p "$tmp/model" "$tmp/cache"
printf '{}\n' > "$tmp/model/config.json"
validate_images() (
  MODEL_DIR="$tmp/model"
  CACHE_DIR="$tmp/cache"
  GPU_DEVICES=0,2
  # shellcheck source=scripts/defaults.sh
  source "$ROOT/scripts/defaults.sh"
  MAX_IMAGES_PER_PROMPT="$1"
  # shellcheck source=scripts/validate-config.sh
  source "$ROOT/scripts/validate-config.sh"
)
validate_images 5
validate_images 16
if validate_images 4 2>/dev/null; then
  echo 'image limit below five was accepted' >&2; exit 1
fi
if validate_images 17 2>/dev/null; then
  echo 'image limit above sixteen was accepted' >&2; exit 1
fi
validate_replayssm() (
  MODEL_DIR="$tmp/model"
  CACHE_DIR="$tmp/cache"
  GPU_DEVICES=0,2
  # shellcheck source=scripts/defaults.sh
  source "$ROOT/scripts/defaults.sh"
  USE_REPLAYSSM="$1"
  # shellcheck source=scripts/validate-config.sh
  source "$ROOT/scripts/validate-config.sh"
)
validate_replayssm 0
if validate_replayssm 1 2>/dev/null; then
  echo 'ReplaySSM enabled value was accepted' >&2; exit 1
fi
replay_marker="$tmp/replayssm-must-not-execute"
malicious_replay="BASH_REMATCH[\$(touch $replay_marker)]"
if validate_replayssm "$malicious_replay" 2>/dev/null; then
  echo 'nonnumeric ReplaySSM value was accepted' >&2; exit 1
fi
[[ ! -e "$replay_marker" ]] || {
  echo 'ReplaySSM validation executed dotenv-controlled syntax' >&2; exit 1
}
validate_mixed_prefill() (
  MODEL_DIR="$tmp/model"
  CACHE_DIR="$tmp/cache"
  GPU_DEVICES=0,2
  # shellcheck source=scripts/defaults.sh
  source "$ROOT/scripts/defaults.sh"
  MIXED_PREFILL_CHUNK="$1"
  # shellcheck source=scripts/validate-config.sh
  source "$ROOT/scripts/validate-config.sh"
)
for policy in off skip 1 128 2048; do validate_mixed_prefill "$policy"; done
# shellcheck disable=SC2016 # literal command substitution is an execution-safety fixture.
for unsafe_policy in '' 0 01 2049 OFF ' skip' '$(touch /tmp/mixed-prefill-must-not-execute)'; do
  if validate_mixed_prefill "$unsafe_policy" 2>/dev/null; then
    echo "unsafe mixed-prefill policy was accepted: $unsafe_policy" >&2; exit 1
  fi
done
[[ ! -e /tmp/mixed-prefill-must-not-execute ]] || { echo 'mixed-prefill validation executed syntax' >&2; exit 1; }
printf 'MIXED_PREFILL_CHUNK=skip\n' > "$tmp/mixed.env"
(
  ENV_FILE="$tmp/mixed.env"
  source "$ROOT/scripts/load-env.sh"
  # shellcheck disable=SC2031 # assignment is deliberately verified inside this subshell.
  [[ "$MIXED_PREFILL_CHUNK" == skip ]]
)
validate_runtime_image() (
  MODEL_DIR="$tmp/model"
  CACHE_DIR="$tmp/cache"
  GPU_DEVICES=0,2
  # shellcheck source=scripts/defaults.sh
  source "$ROOT/scripts/defaults.sh"
  RUNTIME_IMAGE="$1"
  # shellcheck source=scripts/validate-config.sh
  source "$ROOT/scripts/validate-config.sh"
)
base='ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5'
derived='local/glm53-runtime-fixes:d6460a952a88786828a39f44fb99b417144450047dcff446813e4480eb17a8fc'
validate_runtime_image "$base"
validate_runtime_image "$derived"
# shellcheck disable=SC2016 # literal command substitution is an execution-safety fixture.
for unsafe in 'local/glm53-runtime-fixes:latest' 'other/image@sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5' '$(touch /tmp/runtime-image-must-not-execute)'; do
  if validate_runtime_image "$unsafe" 2>/dev/null; then
    echo "unsafe runtime image was accepted: $unsafe" >&2; exit 1
  fi
done
[[ ! -e /tmp/runtime-image-must-not-execute ]] || { echo 'runtime image validation executed syntax' >&2; exit 1; }
printf 'RUNTIME_IMAGE=%s\n' "$derived" > "$tmp/runtime.env"
(
  ENV_FILE="$tmp/runtime.env"
  source "$ROOT/scripts/load-env.sh"
  # shellcheck disable=SC2031 # assignment is deliberately verified inside this subshell.
  [[ "$RUNTIME_IMAGE" == "$derived" ]]
)
echo 'strict dotenv and configuration tests passed'
