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
# shellcheck source=scripts/validate-config.sh
source "$ROOT/scripts/validate-config.sh"
fail=0
for c in docker nvidia-smi python3; do command -v "$c" >/dev/null || { echo "FAIL missing command: $c"; fail=1; }; done
(( fail==0 )) || exit 1
docker info >/dev/null || { echo 'FAIL Docker daemon unavailable'; exit 1; }
[[ -n "${MODEL_DIR:-}" && "$MODEL_DIR" == /* ]] || { echo 'FAIL MODEL_DIR must be absolute'; exit 1; }
[[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/quantization_config.json" ]] || { echo 'FAIL model metadata missing'; exit 1; }
python3 - "$MODEL_DIR" <<'PY'
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]); revision='5ab363a8dcf6405955fd5f99671e01a1c9fb124b'
license_hash='9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a'
license_path=root/'LICENSE'
if not license_path.is_file() or hashlib.sha256(license_path.read_bytes()).hexdigest()!=license_hash:
    raise SystemExit('FAIL checkpoint LICENSE is missing or does not match the pinned revision')
pin=root/'RECIPE_PIN.txt'
if pin.is_file() and pin.read_text().strip()!=f'brandonmusic/GLM-5.3-Flash-tr3-4bpw@{revision}':
    raise SystemExit('FAIL RECIPE_PIN.txt does not identify the tested checkpoint')
metadata=list((root/'.cache/huggingface/download').glob('*.metadata'))
if not pin.is_file():
    commits={p.read_text(errors='replace').splitlines()[0] for p in metadata if p.read_text(errors='replace').splitlines()}
    if commits!={revision}: raise SystemExit(f'FAIL model metadata revisions are not exactly pinned: {sorted(commits)}')
shards=list(root.glob('model-*-of-00120.safetensors'))
size=sum(p.stat().st_size for p in shards)
if len(shards)!=120 or size<170_000_000_000:
    raise SystemExit(f'FAIL incomplete model: shards={len(shards)}, bytes={size}')
print(f'Model revision: {revision}; shards={len(shards)}; bytes={size}; license_sha256={license_hash}')
PY
shards=$(find "$MODEL_DIR" -maxdepth 1 -type f -name 'model-*-of-00120.safetensors' | wc -l)
[[ "$shards" -eq 120 ]] || { echo "FAIL expected 120 model shards, found $shards"; exit 1; }
IFS=',' read -r -a gpus <<< "${GPU_DEVICES:-0,1}"
[[ "${#gpus[@]}" -eq 2 && "${gpus[0]}" != "${gpus[1]}" ]] || { echo 'FAIL select exactly two distinct GPU indices'; exit 1; }
for i in "${gpus[@]}"; do
  row=$(nvidia-smi -i "$i" --query-gpu=index,name,memory.total,pci.bus_id --format=csv,noheader,nounits)
  echo "GPU $row"; name=$(cut -d, -f2 <<<"$row"|xargs); mem=$(cut -d, -f3 <<<"$row"|xargs)
  [[ "$name" == *"RTX PRO 6000 Blackwell"* ]] || { echo "FAIL GPU $i is not RTX PRO 6000 Blackwell"; fail=1; }
  (( mem>=97000 )) || { echo "FAIL GPU $i has ${mem}MiB, expected 96GB model"; fail=1; }
done
python3 - "${gpus[0]}" "${gpus[1]}" <<'PY'
import subprocess,sys
want=[f'GPU{sys.argv[1]}',f'GPU{sys.argv[2]}']; lines=subprocess.check_output(['nvidia-smi','topo','-p2p','r'],text=True).splitlines()
header=lines[0].split(); rows={r.split()[0]:r.split()[1:] for r in lines[1:] if r.strip().startswith('GPU')}
try: status=rows[want[0]][header.index(want[1])]
except Exception as e: raise SystemExit(f'FAIL unable to parse P2P matrix: {e}')
print(f'P2P {want[0]} -> {want[1]}: {status}')
if status!='OK': raise SystemExit('FAIL P2P read status is not OK; see docs/TROUBLESHOOTING.md')
PY
hash=$(sha256sum "$MODEL_DIR/chat_template.jinja"|cut -d' ' -f1)
[[ "$hash" == 0a3c8768eb6b309ef077163b045b4240d5609a9c84ea1b7019aa63d76c9301bc ]] || { echo 'FAIL vision/thinking-control template absent or unknown'; fail=1; }
df -h "$MODEL_DIR"
(( fail==0 )) || exit 1
echo 'PREFLIGHT PASS'
