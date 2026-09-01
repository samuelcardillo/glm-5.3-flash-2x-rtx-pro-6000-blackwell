# Troubleshooting

## Safety rule

Start read-only. Do not copy workstation-specific kernel, IOMMU, driver, Docker-root, or bootloader changes blindly. Preserve rollback and console access before changing boot configuration.

## `At most 0 image(s)` or image HTTP 500

Confirm the container lacks `--language-model-only`, has a nonzero image limit, and the client advertises images. Run the template patch and verifier. The tested revision ships visual weights but a text-only template; without the repair vLLM can report `Failed to apply prompt replacement for mm_items['image'][0]`.

Updated installations require `MAX_IMAGES_PER_PROMPT=5..16`. Legacy values `0..4` now fail intentionally; update the existing `.env` before restarting.

## Startup appears hung

Cold startup took about four minutes because 120 shards load and CUDA/TileLang graphs compile. Watch `journalctl --user -u glm53-2x-rtxpro6000.service -f` and wait for `Application startup complete`. A traceback, exit, OOM, or Xid is failure; compilation progress is not.

## P2P/NCCL hang

Read-only checks:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r
journalctl -k -b | grep -Ei 'AMD-Vi|IOMMU|NVRM|Xid|page fault'
```

The qualified AMD workstation initially showed AMD-Vi I/O page faults during peer DMA. Its host-specific remediation used `iommu=pt` and this NVIDIA override:

```text
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1" NVreg_DmaRemapPeerMmio=1
```

This is evidence, **not a universal recommendation**. Before considering similar changes: capture exact errors/topology; verify vendor guidance; back up boot/modprobe files; ensure console recovery; change one variable; rebuild initramfs only when required; reboot and retest; roll back on new Xids, IOMMU faults, boot failures, or missing GPUs.

## Wrong GPU pair

Set physical host indices in `.env`, e.g. `GPU_DEVICES=0,2`. Inside the container they become logical devices 0 and 1. Never assume contiguous indices on mixed-GPU workstations.

## OOM or insufficient KV

Do not raise memory utilization above 0.950 first. Check other processes, exact checkpoint, template mode, image count, and graph settings. Vision reduced the qualified pool from 734,003 to 545,259 while preserving 262K context.

## Docker storage fills root

Use your distribution's documented stopped-service migration with backup. Do not copy a live/inconsistent image graph; that previously produced containers missing `/opt/venv/bin/vllm`.

## Random-looking words, repeated text, or reasoning until length

First separate raw `reasoning_content` from final `content`. The original checkpoint template ignored `enable_thinking=false` and always opened a maximum-effort reasoning block. On a large synthetic design task, that consumed all 4,096 completion tokens, returned `finish_reason=length`, and produced no final content. Clients that display or concatenate the reasoning stream can therefore look stuck even when transport is healthy.

Update the recipe, rerun `scripts/apply-vision-template.py "$MODEL_DIR"`, and restart the service. The launcher sets a server-side `enable_thinking=false` default. To opt in explicitly, send:

```json
{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}
```

Also set `USE_REPLAYSSM=0`. A matched temperature-0, 32K-token, concurrency-4 regression found 4/40 repeated final-content streams with thinking off and 7/40 `handlehandle...` reasoning loops with thinking at maximum while ReplaySSM was enabled. A unique-prefix phase then hit `ValueError: ReplaySSM prefill source/state row count mismatch`, killed EngineCore, and failed 36/40 requests. With only ReplaySSM disabled, the corresponding 120 requests completed with zero genuine loops, zero request errors, and no engine death. Adaptive MTP remains enabled through its standard full-state rollback path.

For an existing user-systemd installation, edit the repository `.env`, rerun `scripts/install-user-service.sh`, and then run `systemctl --user restart glm53-2x-rtxpro6000.service`. The installer refreshes the service's private environment copy, renders the updated unit with `Restart=always`, and runs `systemctl --user daemon-reload`; editing only the repository `.env` does not update an already installed service.

For diagnosis, retain the raw SSE stream and record prompt/completion token counts and `finish_reason`. Do not treat temperature 1 or repetition penalties as deterministic controls. The raw evidence above came directly from server SSE; it was not client concatenation. `Restart=always` is defense in depth for engine exits, not a substitute for disabling the corrupt path.

See [the September 2026 investigation](2026-09-01-corruption-investigation.md) for the controlled evidence and exact scope.
