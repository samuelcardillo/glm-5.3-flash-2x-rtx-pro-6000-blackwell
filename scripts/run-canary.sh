#!/usr/bin/env bash
# Run a disruptive loopback canary and restore the user service on every exit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# -ge 2 && "$2" == -- ]] || { echo "usage: $0 CANDIDATE_ENV -- CHECK [ARG ...]" >&2; exit 2; }
CANDIDATE_ENV="$1"; shift 2
[[ $# -gt 0 ]] || { echo 'at least one check command is required' >&2; exit 2; }
CONTROL_ENV="${CONTROL_ENV:-$ROOT/.env}"
SERVICE_UNIT="${SERVICE_UNIT:-glm53-longctx-mtp.service}"
SYSTEMCTL_BIN="${SYSTEMCTL_BIN:-systemctl}"
if [[ "$SYSTEMCTL_BIN" == systemctl ]]; then
  XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path:${XDG_RUNTIME_DIR}/bus}"
  export XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS
fi
DOCKER_BIN="${DOCKER_BIN:-docker}"
SERVE_SCRIPT="${SERVE_SCRIPT:-$ROOT/scripts/serve.sh}"
WAIT_SCRIPT="${WAIT_SCRIPT:-$ROOT/scripts/wait-ready.py}"
LOAD_ENV_SCRIPT="${LOAD_ENV_SCRIPT:-$ROOT/scripts/load-env.sh}"
DEFAULTS_SCRIPT="${DEFAULTS_SCRIPT:-$ROOT/scripts/defaults.sh}"
VALIDATE_SCRIPT="${VALIDATE_SCRIPT:-$ROOT/scripts/validate-config.sh}"
CANARY_ARTIFACT_DIR="${CANARY_ARTIFACT_DIR:-}"
[[ -f "$CANDIDATE_ENV" && -f "$CONTROL_ENV" ]] || { echo 'candidate/control env missing' >&2; exit 2; }

read_candidate_profile() {
  local env_file="$1" output_file="$2"
  (
    ENV_FILE="$env_file"; export ENV_FILE
    # shellcheck source=scripts/load-env.sh
    source "$LOAD_ENV_SCRIPT"
    # shellcheck source=scripts/defaults.sh
    source "$DEFAULTS_SCRIPT"
    # shellcheck source=scripts/validate-config.sh
    source "$VALIDATE_SCRIPT"
    printf '%s\0' "$BIND_ADDRESS" "$PORT" "$CONTAINER_NAME" "$SERVED_MODEL_NAME" "$MAX_MODEL_LEN" "$DFLASH_TOKENS"
  ) >"$output_file"
}

read_control_profile() {
  local env_file="$1" output_file="$2"
  python3 - "$env_file" "$output_file" <<'PY'
from pathlib import Path
import re, sys

source, destination = map(Path, sys.argv[1:])
values = {}
for raw in source.read_text().splitlines():
    if not raw or raw.startswith('#'):
        continue
    if '=' not in raw:
        raise SystemExit('invalid control dotenv line')
    key, value = raw.split('=', 1)
    if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) or key in values:
        raise SystemExit('invalid or duplicate control dotenv key')
    values[key] = value
readiness_host = values.get('BIND_ADDRESS', '127.0.0.1')
if readiness_host == '0.0.0.0':
    readiness_host = '127.0.0.1'
elif readiness_host in ('::', '::1'):
    readiness_host = '[::1]'
fields = (
    readiness_host,
    values.get('PORT', '8000'),
    values.get('CONTAINER_NAME', 'glm53-longctx-mtp'),
    values.get('SERVED_MODEL_NAME', 'overlord-testing'),
    values.get('MAX_MODEL_LEN', '262144'),
    'legacy-control',
)
if not re.fullmatch(r'[0-9]+', fields[1]) or not re.fullmatch(r'[0-9]+', fields[4]):
    raise SystemExit('invalid control numeric profile')
destination.write_bytes(b'\0'.join(x.encode() for x in fields) + b'\0')
PY
}

tmp_dir=$(mktemp -d)
candidate_profile="$tmp_dir/candidate.profile"
control_profile="$tmp_dir/control.profile"
# Both profiles are fully validated before the service is touched.
read_candidate_profile "$CANDIDATE_ENV" "$candidate_profile"
read_control_profile "$CONTROL_ENV" "$control_profile"
mapfile -d '' -t candidate <"$candidate_profile"
mapfile -d '' -t control <"$control_profile"
[[ ${#candidate[@]} -eq 6 && ${#control[@]} -eq 6 ]] || { echo 'invalid profile capture' >&2; exit 2; }
[[ "${candidate[0]}" == 127.0.0.1 || "${candidate[0]}" == localhost || "${candidate[0]}" == ::1 ]] || { echo 'candidate must bind to loopback' >&2; exit 2; }
[[ "${candidate[5]}" == 5 ]] || { echo 'Candidate must use the qualified DFlash2 K5 profile' >&2; exit 2; }
[[ "${candidate[2]}" != "${control[2]}" ]] || { echo 'candidate and control container names must differ' >&2; exit 2; }
"$SYSTEMCTL_BIN" --user is-active --quiet "$SERVICE_UNIT" || { echo 'control service must be active before canary' >&2; exit 2; }

candidate_pid=''
service_stopped=0
restoring=0
cleanup() {
  local status=$? restore_status=0
  (( restoring == 0 )) || exit "$status"
  restoring=1
  trap - EXIT INT TERM HUP
  set +e
  if [[ -n "$candidate_pid" ]]; then
    kill -TERM -- "-$candidate_pid" >/dev/null 2>&1
    wait "$candidate_pid" >/dev/null 2>&1
  fi
  "$DOCKER_BIN" rm -f "${candidate[2]}" >/dev/null 2>&1
  if [[ -n "$CANARY_ARTIFACT_DIR" && -f "$tmp_dir/candidate.log" ]]; then
    umask 077
    mkdir -p "$CANARY_ARTIFACT_DIR"
    cp "$tmp_dir/candidate.log" "$CANARY_ARTIFACT_DIR/candidate.log"
    chmod 0600 "$CANARY_ARTIFACT_DIR/candidate.log"
  fi
  if (( service_stopped == 1 )); then
    "$SYSTEMCTL_BIN" --user start "$SERVICE_UNIT" || restore_status=1
    if (( restore_status == 0 )); then
      "$WAIT_SCRIPT" --base-url "http://${control[0]}:${control[1]}" --model "${control[3]}" --context "${control[4]}" --timeout 900 --container "${control[2]}" || restore_status=1
    fi
  fi
  rm -rf "$tmp_dir"
  if (( restore_status != 0 )); then
    echo 'RESTORATION FAILED: control service or endpoint did not recover' >&2
    exit 125
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# Arm restoration before stop: systemctl may return nonzero after a partial stop.
service_stopped=1
"$SYSTEMCTL_BIN" --user stop "$SERVICE_UNIT"
setsid env ENV_FILE="$CANDIDATE_ENV" "$SERVE_SCRIPT" >"$tmp_dir/candidate.log" 2>&1 &
# shellcheck disable=SC2031 # $! is intentionally captured in the parent shell.
candidate_pid=$!
"$WAIT_SCRIPT" --base-url "http://${candidate[0]}:${candidate[1]}" --model "${candidate[3]}" --context "${candidate[4]}" --timeout 1800 --container "${candidate[2]}" --process-pid "$candidate_pid"
"$@"
printf 'CANARY_CHECKS_PASSED\n'
