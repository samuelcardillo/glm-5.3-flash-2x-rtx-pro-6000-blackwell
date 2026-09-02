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
for command_name in docker nvidia-smi python3; do
  command -v "$command_name" >/dev/null || { echo "FAIL missing command: $command_name"; exit 1; }
done
docker info >/dev/null || { echo 'FAIL Docker daemon unavailable'; exit 1; }
docker image inspect "$RUNTIME_IMAGE" >/dev/null || { echo 'FAIL immutable v0.6 runtime image is not present'; exit 1; }
python3 - "$MODEL_DIR" "$DRAFT_DIR" "$DFLASH_TOKENS" <<'PY'
import json
from pathlib import Path, PurePosixPath
import sys

target, draft = map(Path, sys.argv[1:3])
tokens = int(sys.argv[3])
target_pin = 'wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4'
draft_pin = 'incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410'
for root, expected in ((target, target_pin), (draft, draft_pin)):
    pin = root / 'RECIPE_PIN.txt'
    if not pin.is_file() or pin.read_text().strip() != expected:
        raise SystemExit(f'FAIL missing or incorrect immutable pin: {root}')
index_path = target / 'model.safetensors.index.json'
try:
    index = json.loads(index_path.read_text(encoding='utf-8'))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f'FAIL invalid checkpoint index: {exc}')
weight_map = index.get('weight_map')
if not isinstance(weight_map, dict) or not weight_map:
    raise SystemExit('FAIL target checkpoint index has no weight_map')
shards = sorted(set(weight_map.values()))
if len(shards) != 16:
    raise SystemExit(f'FAIL expected 16 target shards, found {len(shards)}')
size = 0
for name in shards:
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or not name:
        raise SystemExit(f'FAIL unsafe shard path: {name!r}')
    path = target / relative
    if not path.is_file():
        raise SystemExit(f'FAIL target shard missing: {name}')
    size += path.stat().st_size
if size != 136_686_260_192:
    raise SystemExit(f'FAIL target checkpoint byte total mismatch: {size}')
for required in ('config.json', 'quantization_config.json', 'processor_config.json', 'tokenizer_config.json'):
    if not (target / required).is_file():
        raise SystemExit(f'FAIL target metadata missing: {required}')
draft_model = draft / 'model.safetensors'
if not draft_model.is_file() or draft_model.stat().st_size != 2_342_169_800:
    raise SystemExit('FAIL DFlash2 weights are missing or have the wrong byte size')
config = json.loads((draft / 'config.json').read_text(encoding='utf-8'))
if config.get('architectures') != ['DFlash2DraftModel']:
    raise SystemExit('FAIL draft checkpoint is not DFlash2DraftModel')
dflash = config.get('dflash_config') or {}
block_size = int(dflash.get('block_size', 0))
if block_size != 8 or not 1 <= tokens < block_size:
    raise SystemExit(f'FAIL incompatible DFlash2 block/tokens: {block_size}/{tokens}')
if not dflash.get('target_layer_ids') or dflash.get('mask_token_id') is None:
    raise SystemExit('FAIL DFlash2 target taps or mask token missing')
print(f'Model artifacts: target_shards={len(shards)}, target_bytes={size}, DFlash2=K{tokens}')
PY
IFS=',' read -r -a gpus <<< "$GPU_DEVICES"
[[ "${#gpus[@]}" -eq 2 && "${gpus[0]}" != "${gpus[1]}" ]] || { echo 'FAIL select exactly two distinct GPU indices'; exit 1; }
for index in "${gpus[@]}"; do
  row=$(nvidia-smi -i "$index" --query-gpu=index,name,memory.total --format=csv,noheader,nounits)
  name=$(cut -d, -f2 <<<"$row" | xargs)
  memory=$(cut -d, -f3 <<<"$row" | xargs)
  [[ "$name" == *"RTX PRO 6000 Blackwell"* ]] || { echo "FAIL GPU $index is not RTX PRO 6000 Blackwell"; exit 1; }
  (( memory>=97000 )) || { echo "FAIL GPU $index has ${memory}MiB, expected 96GB"; exit 1; }
done
python3 - "${gpus[0]}" "${gpus[1]}" <<'PY'
import subprocess, sys
want = [f'GPU{sys.argv[1]}', f'GPU{sys.argv[2]}']
lines = subprocess.check_output(['nvidia-smi', 'topo', '-p2p', 'r'], text=True).splitlines()
header = lines[0].split()
rows = {row.split()[0]: row.split()[1:] for row in lines[1:] if row.strip().startswith('GPU')}
try:
    status = rows[want[0]][header.index(want[1])]
except Exception as exc:
    raise SystemExit(f'FAIL unable to parse P2P matrix: {exc}')
if status != 'OK':
    raise SystemExit('FAIL P2P read status is not OK')
print('Selected GPU pair and P2P access: pass')
PY
echo 'PREFLIGHT PASS'
