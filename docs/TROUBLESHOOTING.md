# Troubleshooting

## Safety rule

Start read-only. Do not copy workstation-specific kernel, IOMMU, driver, Docker-root, power-limit, or bootloader changes blindly. Preserve rollback and console access before changing host configuration.

## Startup appears hung

The v0.6 runtime deliberately compiles and warms many CUDA/TileLang shapes before becoming release-ready. `/health` may respond before this finishes. Use Docker health or `scripts/wait-ready.py`; do not send qualification traffic until the container is `healthy`.

Compilation progress is normal. A traceback, container exit, OOM, Xid, or `unhealthy` state is failure.

## Wrong GPU pair

Set exactly two physical host indices in `.env`, for example `GPU_DEVICES=0,2`. The launcher quotes the comma-separated Docker device request correctly. Inside the container, selected devices become logical GPUs 0 and 1. Never use `--gpus all` on a mixed-GPU host.

## P2P/NCCL problems

Read-only checks:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
journalctl -k -b | grep -Ei 'IOMMU|NVRM|Xid|page fault'
```

The public preflight requires P2P read status `OK`. Kernel/module changes are host-specific and must be researched, backed up, applied one variable at a time, and rolled back on new faults.

## OOM or insufficient KV

Confirm the exact target, draft, image digest and profile before changing memory limits. Remove unrelated GPU processes. The qualified profile uses 0.950 memory utilization and reports 2,926,692 KV tokens. Raising utilization can remove startup safety margin and requires a new full qualification.

## Thinking leaks into final content

The pinned K3 template always opens a reasoning block and does not natively honor `enable_thinking=false`. The launcher automatically derives and mounts a source-hash-pinned template from `MODEL_DIR/chat_template.jinja`; it does not mutate the checkpoint.

Expected hashes:

- source: `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`
- derived: `5bcdf9be4e5b4a6cf2017f74f7e0b5c7f91bb814a275438dc678dd48da1f81b5`

If the launcher rejects the source, do not bypass the check. Verify the pinned target revision. To opt into thinking explicitly:

```json
{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}
```

## A 1M test ends with `finish_reason=length`

That is an output-budget failure, not proof of successful retrieval. The first local attempt used 128 output tokens and was rejected for this reason. Use the documented 512-token budget and require six-of-six records, exactly 1,000,000 prompt tokens, `[DONE]`, and `finish_reason=stop`.

## Image failures

Confirm `MAX_IMAGES_PER_PROMPT=16`, `MAX_VIDEOS_PER_PROMPT=0`, the exact K3 processor files, and the mounted derived template. Run `scripts/verify-vision-limit.py`; HTTP 200 alone is insufficient because the test requires exact ordered recognition of all 16 images and rejection of image 17.

## DFlash2 does not start

Confirm:

- the draft revision and `RECIPE_PIN.txt`;
- architecture `DFlash2DraftModel`;
- `DFLASH_TOKENS=5`;
- `DFLASH_KV_CACHE_DTYPE=bfloat16`;
- `ACCEPT_DFLASH2_RESEARCH_LICENSE=1` after reviewing the research/evaluation terms.

DFlash2 is not licensed for unrestricted commercial use. Do not bypass the acknowledgement gate.

## Docker storage fills a filesystem

The target is 127.30GiB before image/cache headroom. Select destinations with sufficient free space before download. Move stopped, internally consistent cache trees only; do not copy a live model download or Docker graph.

## Service migration

Old Brandon/MTP `.env` files are intentionally incompatible with this profile. Create a new configuration from `config/example.env`, review the DFlash2 terms, run preflight, reinstall the user service, then restart it. Editing the repository `.env` alone does not update an already installed private service environment.

The canary wrapper traps ordinary exit and `INT`, `TERM`, and `HUP`, then removes the candidate and verifies restoration. It cannot protect against `SIGKILL` or host power loss.