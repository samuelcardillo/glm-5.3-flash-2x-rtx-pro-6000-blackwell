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
echo 'strict dotenv and configuration tests passed'
