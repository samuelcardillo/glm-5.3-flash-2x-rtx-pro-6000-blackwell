# GLM-5.3 Flash on 2× RTX PRO 6000 Blackwell 96GB

A reproducible, vision-enabled, 262K-context deployment recipe for the EXL3/TR3 4-bpw GLM-5.3 Flash checkpoint on two PCIe-connected NVIDIA RTX PRO 6000 Blackwell 96GB workstation GPUs.

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Read this first

- This is an **integration recipe**, not a new model, quantization method, or serving engine.
- The checkpoint was created by **Brandon M. Music** using ShapleyMCG/TR3. It is source-available under a non-OSI license; review its terms before use.
- The dual-GPU runtime and pinned container were built by **T.J. Purtell and upstream contributors**.
- No model weights or container layers are stored here.
- This targets **RTX PRO 6000 Blackwell 96GB**, not RTX 6000 Ada 48GB or RTX A6000 48GB.
- The validated checkpoint is approximately 175.6GB; two 48GB cards cannot run this profile unchanged.

See [ATTRIBUTIONS.md](ATTRIBUTIONS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [PROVENANCE.md](PROVENANCE.md). The notices document two unresolved source/build-provenance gaps in third-party binary images; do not mirror or redistribute the composed image as fully source-audited.

## Validated profile

- 2× RTX PRO 6000 Blackwell 96GB, including a mixed Max-Q/full-workstation pair
- Explicit physical GPU selection on a host also containing an RTX 5090
- Pinned checkpoint and pinned GHCR image
- TP2 + DCP2 over PCIe
- Adaptive MTP K1–K5 with ReplaySSM
- NVFP4 MLA KV cache and prefix caching
- 262,144-token request ceiling
- Up to 16 images per prompt; video disabled
- OpenAI-compatible text, structured tools, and semantic image input
- Exact 128K and 261.9K retrieval acceptance tests
- Zcode image → GLM vision → `Write` tool workflow
- Login-persistent user systemd service, with optional administrator-enabled linger for boot-time start

## Requirements

Validated with Linux 6.17/Ubuntu 24.04-family userspace, NVIDIA driver 590.48.01, Docker 29.1.3, and two 97,887MiB RTX PRO 6000 Blackwell cards. You also need NVIDIA Container Toolkit, the Hugging Face `hf` CLI, healthy CUDA peer access, and roughly 220GB free storage for checkpoint, image, cache, and headroom.

The recipe does not automatically modify the kernel, bootloader, NVIDIA module parameters, Docker root, firewall, or network exposure.

## Quick start

### 1. Clone and configure

```bash
git clone https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell.git
cd glm-5.3-flash-2x-rtx-pro-6000-blackwell
cp config/example.env .env
$EDITOR .env
```

Set absolute `MODEL_DIR` and `CACHE_DIR` paths and two physical GPU indices. The safe default binds the API to `127.0.0.1`; set `BIND_ADDRESS` to a specific trusted LAN/Tailnet address only when remote clients need direct access. For cards at physical indices 0 and 2:

```text
GPU_DEVICES=0,2
```

### 2. Review the checkpoint license and download the tested revision

Read the [checkpoint model card](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw), its [license at the tested revision](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/blob/5ab363a8dcf6405955fd5f99671e01a1c9fb124b/LICENSE), and the [retained local copy](THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt).

If its terms apply to you:

```bash
I_ACCEPT_SHAPLEYMCG_LICENSE=yes \
  scripts/download-model.sh /absolute/path/to/GLM-5.3-Flash-tr3-4bpw
```

This pins `brandonmusic/GLM-5.3-Flash-tr3-4bpw@5ab363a8dcf6405955fd5f99671e01a1c9fb124b`.

### 3. Apply the reversible vision-template repair

The tested checkpoint contains visual weights and processors but ships a template that converts media into a text-only reminder:

```bash
scripts/apply-vision-template.py "$MODEL_DIR"
```

The patch refuses unknown revisions and keeps `chat_template.text-only.bak.jinja`. Restore with:

```bash
scripts/apply-vision-template.py --restore "$MODEL_DIR"
```

### 4. Preflight

```bash
scripts/preflight.sh
```

This read-only check validates the GPU class/memory, two-card selection, 120 shards, template checksum, Docker access, and CUDA P2P read status.

### 5. Launch

Foreground:

```bash
scripts/serve.sh
```

Persistent user service:

```bash
scripts/install-user-service.sh
systemctl --user start glm53-2x-rtxpro6000.service
journalctl --user -u glm53-2x-rtxpro6000.service -f
```

The enabled user service starts when your user systemd manager starts. If it must start before interactive login, an administrator can explicitly enable lingering with `sudo loginctl enable-linger "$USER"`; review that policy change before applying it.

Cold startup can take about four minutes. Wait for `Application startup complete`.

### 6. Verify text, tools, and actual pixels

```bash
python3 scripts/verify.py \
  --base-url http://127.0.0.1:8000 \
  --model glm-5.3-flash-local
```

The standard-library-only verifier generates a PNG containing `73`, then checks health, exact text, structured tool arguments, and semantic image recognition.

Reproduce the exact-token retrieval checks separately:

```bash
python3 scripts/verify-long-context.py \
  --base-url http://127.0.0.1:8000 \
  --model glm-5.3-flash-local \
  --targets 128000,261900
```

This uses the live server's `/tokenize` endpoint to construct exact chat-prompt lengths, then requires both the server-reported prompt count and retrieved needle to match.

## API exposure

The endpoint has no authentication and binds to loopback by default. For remote access, prefer a specific trusted LAN/Tailnet `BIND_ADDRESS` or place an authenticated gateway such as LiteLLM in front. Avoid `0.0.0.0` unless a firewall controls port 8000; never expose it directly to the Internet.

## Vision capacity tradeoff

| Mode | Observed KV pool | Full 262K concurrency estimate |
|---|---:|---:|
| Text-only (`--language-model-only`) | 734,003 tokens | 2.80× |
| Vision, 16 images, video disabled | 545,259 tokens | 2.08× |

Vision loads approximately 1.05GiB of BF16 visual tensors. The validated 262,144-token maximum remains intact.

## Zcode

Merge [examples/zcode-config.fragment.json](examples/zcode-config.fragment.json), replace `SERVER_LAN_IP`, and fully restart Zcode. Keep the explicit reasoning-off profile: raising output tokens or timeouts only prolongs a reasoning loop.

## Operations

```bash
systemctl --user status glm53-2x-rtxpro6000.service
systemctl --user restart glm53-2x-rtxpro6000.service
systemctl --user stop glm53-2x-rtxpro6000.service
docker logs glm53-flash-2x-rtxpro6000
```

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) before changing IOMMU, P2P, driver, or Docker storage settings.

## Reproducibility policy

Pin every model, image, runtime, template, and benchmark revision. Keep text-only and multimodal measurements separate. State metric boundaries and hardware topology. Never commit credentials, model files, Docker layers, or machine-specific secrets.

```bash
scripts/ci.sh
```

## Credits and licenses

This recipe relies on Brandon M. Music, Z.ai, T.J. Purtell, Luke Alonso, Local Inference Lab and B12x/SparkInfer contributors, Turboderp and ExLlamaV3 contributors, Johnny-Liou, CZT0, ZJY0516, Jared (as named by the runtime author), vLLM contributors, cstechdev, MiaAI-Lab contributors, Hugging Face, PyTorch, NVIDIA, Docker/Moby, and other upstream contributors. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for exact contributions, immutable links, licenses, and unresolved gaps.

Original recipe scripts and documentation are Apache-2.0. The checkpoint, runtime, containers, drivers, CUDA stack, and dependencies retain their own licenses. In particular, the ShapleyMCG checkpoint is source-available under its own non-OSI terms.
