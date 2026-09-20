#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=scripts/load-env.sh
  source "$ROOT/scripts/load-env.sh"
fi
# shellcheck source=scripts/defaults.sh
source "$ROOT/scripts/defaults.sh"
mkdir -p "$CACHE_DIR"
# shellcheck source=scripts/container-lifecycle.sh
source "$ROOT/scripts/container-lifecycle.sh"
stop_owned_container
