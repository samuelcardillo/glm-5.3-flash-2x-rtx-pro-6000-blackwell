#!/usr/bin/env bash
# Shared, bounded ownership checks for the systemd-managed GLM container.

MANAGED_LABEL_KEY="ai.nous.glm53.owner"
MANAGED_LABEL_VALUE="glm53-2x-rtxpro6000"
RUNTIME_STATE_DIR="${XDG_RUNTIME_DIR:-${CACHE_DIR:?CACHE_DIR is required}/runtime}"
CID_FILE="${RUNTIME_STATE_DIR}/${CONTAINER_NAME}.cid"
mkdir -p "$RUNTIME_STATE_DIR"
chmod 700 "$RUNTIME_STATE_DIR" 2>/dev/null || true

_docker_bounded() {
  local seconds=$1
  shift
  timeout --foreground --kill-after=5s "${seconds}s" docker "$@"
}

_container_label() {
  _docker_bounded 10 inspect -f "{{ index .Config.Labels \"${MANAGED_LABEL_KEY}\" }}" "$1" 2>/dev/null
}

_container_name() {
  local name
  name=$(_docker_bounded 10 inspect -f '{{.Name}}' "$1" 2>/dev/null) || return 1
  printf '%s\n' "${name#/}"
}

_owned_container_by_name() {
  local id label
  id=$(_docker_bounded 10 inspect -f '{{.Id}}' "$CONTAINER_NAME" 2>/dev/null) || return 1
  label=$(_container_label "$id") || return 2
  if [[ "$label" != "$MANAGED_LABEL_VALUE" ]]; then
    echo "Refusing unmanaged container name collision: $CONTAINER_NAME" >&2
    return 2
  fi
  printf '%s\n' "$id"
}

_owned_container_id() {
  local id label name
  if [[ -s "$CID_FILE" ]]; then
    IFS= read -r id < "$CID_FILE"
    if [[ "$id" =~ ^[a-f0-9]{12,64}$ ]]; then
      label=$(_container_label "$id") || label=""
      name=$(_container_name "$id") || name=""
      if [[ "$label" == "$MANAGED_LABEL_VALUE" && "$name" == "$CONTAINER_NAME" ]]; then
        printf '%s\n' "$id"
        return 0
      fi
    fi
  fi
  _owned_container_by_name
}

stop_owned_container() {
  local id running status
  if id=$(_owned_container_id); then
    :
  else
    status=$?
    if (( status == 2 )); then
      return 1
    fi
    rm -f -- "$CID_FILE"
    return 0
  fi

  if ! _docker_bounded 40 stop --time 30 "$id" >/dev/null 2>&1; then
    _docker_bounded 15 kill "$id" >/dev/null 2>&1 || {
      echo "Failed to stop or kill managed container $id" >&2
      return 1
    }
  fi

  running=$(_docker_bounded 10 inspect -f '{{.State.Running}}' "$id" 2>/dev/null) || running="absent"
  if [[ "$running" == true ]]; then
    echo "Managed container is still running after stop: $id" >&2
    return 1
  fi
  if _docker_bounded 10 inspect "$id" >/dev/null 2>&1; then
    _docker_bounded 15 rm -f "$id" >/dev/null 2>&1 || {
      echo "Failed to remove stopped managed container $id" >&2
      return 1
    }
    if _docker_bounded 10 inspect "$id" >/dev/null 2>&1; then
      echo "Managed container still exists after removal: $id" >&2
      return 1
    fi
  fi
  rm -f -- "$CID_FILE"
  return 0
}

prepare_owned_container() {
  if _docker_bounded 10 inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    _owned_container_by_name >/dev/null || return 1
    stop_owned_container
  else
    rm -f -- "$CID_FILE"
  fi
}
