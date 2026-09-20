#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/bin" "$tmp/cache" "$tmp/runtime"
log="$tmp/docker.log"

cat > "$tmp/bin/timeout" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) shift ;;
    --kill-after=*) shift ;;
    *s) shift; break ;;
    *) break ;;
  esac
done
exec "$@"
SH

cat > "$tmp/bin/docker" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$DOCKER_LOG"
id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
stale=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
case "$*" in
  "inspect test-container") exit 0 ;;
  "inspect $id") [[ ! -e "$REMOVED_FILE" ]] ;;
  "inspect $stale") exit 0 ;;
  "inspect -f {{.Id}} test-container") printf '%s\n' "$id" ;;
  "inspect -f {{ index .Config.Labels \"ai.nous.glm53.owner\" }} $id"|\
  "inspect -f {{ index .Config.Labels \"ai.nous.glm53.owner\" }} $stale")
    printf '%s\n' "${FAKE_LABEL:-unmanaged}" ;;
  "inspect -f {{.Name}} $id") printf '/test-container\n' ;;
  "inspect -f {{.Name}} $stale") printf '/other-container\n' ;;
  "inspect -f {{.State.Running}} $id") printf 'false\n' ;;
  "inspect -f {{.State.Running}} $stale") printf 'false\n' ;;
  "stop --time 30 $id")
    [[ "${FAKE_STOP_FAIL:-0}" == 0 ]] || exit 1 ;;
  "kill $id") exit 0 ;;
  "rm -f $id") : > "$REMOVED_FILE" ;;
  *) echo "unexpected fake docker invocation: $*" >&2; exit 3 ;;
esac
SH
chmod +x "$tmp/bin/timeout" "$tmp/bin/docker"

export PATH="$tmp/bin:$PATH"
export DOCKER_LOG="$log"
export REMOVED_FILE="$tmp/removed"
export CACHE_DIR="$tmp/cache"
export XDG_RUNTIME_DIR="$tmp/runtime"
export CONTAINER_NAME=test-container
# shellcheck source=scripts/container-lifecycle.sh
source "$ROOT/scripts/container-lifecycle.sh"

export FAKE_LABEL=unmanaged
if prepare_owned_container 2>/dev/null; then
  echo 'unmanaged name collision was accepted' >&2
  exit 1
fi
! grep -q '^stop ' "$log"

: > "$log"
rm -f "$REMOVED_FILE"
export FAKE_LABEL=glm53-2x-rtxpro6000
prepare_owned_container
id=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
grep -q "stop --time 30 $id" "$log"
grep -q "rm -f $id" "$log"

: > "$log"
rm -f "$REMOVED_FILE"
stale=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
printf '%s\n' "$stale" > "$CID_FILE"
stop_owned_container
! grep -q "stop --time 30 $stale" "$log"
grep -q "stop --time 30 $id" "$log"

: > "$log"
rm -f "$REMOVED_FILE"
export FAKE_STOP_FAIL=1
printf '%s\n' "$id" > "$CID_FILE"
stop_owned_container
grep -q "kill $id" "$log"

echo 'container lifecycle ownership and fallback tests passed'
