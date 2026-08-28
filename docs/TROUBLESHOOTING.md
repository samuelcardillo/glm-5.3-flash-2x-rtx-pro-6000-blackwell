# Troubleshooting

## Safety rule

Start read-only. Do not copy workstation-specific kernel, IOMMU, driver, Docker-root, or bootloader changes blindly. Preserve rollback and console access before changing boot configuration.

## `At most 0 image(s)` or image HTTP 500

Confirm the container lacks `--language-model-only`, has a nonzero image limit, and the client advertises images. Run the template patch and verifier. The tested revision ships visual weights but a text-only template; without the repair vLLM can report `Failed to apply prompt replacement for mm_items['image'][0]`.

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

## Zcode reasons until length

Keep the explicit `off` profile. Larger timeouts/output budgets only prolong the loop; they do not create tool transition.
