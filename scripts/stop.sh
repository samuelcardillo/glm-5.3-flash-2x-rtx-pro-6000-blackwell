#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-${ROOT}/.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck source=scripts/load-env.sh
  source "$ROOT/scripts/load-env.sh"
fi
CONTAINER_NAME="${CONTAINER_NAME:-glm53-flash-2x-rtxpro6000}"
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker stop --timeout 30 "$CONTAINER_NAME"
fi
