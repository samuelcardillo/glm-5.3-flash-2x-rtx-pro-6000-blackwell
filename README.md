# GLM-5.3 Flash K3 + DFlash2 on 2× RTX PRO 6000 Blackwell 96GB

A revision-pinned, OpenAI-compatible recipe for one-million-token context, 16-image prompts, and high-throughput DFlash2 speculative decoding on two PCIe-connected RTX PRO 6000 Blackwell 96GB GPUs.

This repository integrates and safety-hardens T.J. Purtell's v0.6 runtime. It does not redistribute model weights or container layers.

## Important license boundary

The DFlash2 draft checkpoint is licensed **CC BY-NC-ND 4.0 for research and evaluation**. Commercial use requires separate permission from Inco AI. The download and launch paths fail closed until `ACCEPT_DFLASH2_RESEARCH_LICENSE=1` is set after reviewing the terms.

See [ATTRIBUTIONS.md](ATTRIBUTIONS.md), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and [PROVENANCE.md](PROVENANCE.md).

## Locally qualified result

Qualified on a mixed pair consisting of one RTX PRO 6000 Blackwell Max-Q 96GB and one full RTX PRO 6000 Blackwell Workstation Edition 96GB. An unrelated RTX 5090 was excluded explicitly.

- Target: `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4`
- Draft: `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
- Runtime: tpurtell v0.6 image pinned by OCI digest
- TP2 + EP2 + DCP2, B12x sparse MLA, FP8 target KV, replicated BF16 DFlash2 K5 draft KV
- 1,048,576-token request ceiling
- 2,926,692-token reported KV pool, or 2.79 request-equivalents at the maximum context
- 16 scheduler slots
- 16 images accepted in one request; image 17 rejected; video disabled
- Thinking off by default, with explicit opt-in retained
- ReplaySSM absent
- Exact one-million-token six-needle retrieval: 6/6 in 284.859 seconds
- Repetition regression: 80/80 requests, zero loops and zero errors
- Seven-case semantic content suite: 7/7 pass; 39.30% aggregate DFlash acceptance
- Control deployment restored and verified after the disruptive canary

### Local decode throughput

Five measured runs per point after two warmups, 256 output tokens per sequence, fixed code-agent fixture and seed. Throughput excludes TTFT and sums each sequence's first-to-last-token decode rate.

| Concurrency | Aggregate decode | Median DFlash acceptance |
|---:|---:|---:|
| 1 | 179.50 tok/s | 64.92% |
| 2 | 317.07 tok/s | 65.45% |
| 4 | 493.40 tok/s | 65.97% |
| 8 | 759.72 tok/s | 66.32% |
| 16 | 1,045.47 tok/s | 66.46% |

T.J. Purtell's final v0.6 receipt reports 222.6 tok/s at C1 and 1,067.2 tok/s at C16 on two cards explicitly capped at 400 W each. This host cannot reproduce that power condition: its Max-Q card was at 300 W and cannot exceed 325 W, while the full card was at 600 W. The local C16 result is 2.0% below that final upstream receipt and 4.0% above upstream's initially selected 1,004.9 tok/s profile. Do not transfer performance claims between machines.

### Local prefill

Exact unique prompts, three runs each; client request-to-first-token timing includes server tokenization and one-token handoff.

| Prompt tokens | Median effective prefill | Median TTFT |
|---:|---:|---:|
| 8,192 | 4,258.74 tok/s | 1.924 s |
| 16,384 | 4,107.44 tok/s | 3.989 s |
| 32,768 | 4,219.75 tok/s | 7.765 s |
| 65,536 | 4,230.59 tok/s | 15.491 s |
| 128,000 | 4,199.66 tok/s | 30.479 s |

Detailed methodology is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Requirements

- Linux with Docker and NVIDIA Container Toolkit
- Two 96GB RTX PRO 6000 Blackwell GPUs with CUDA P2P read access
- NVIDIA driver compatible with the pinned CUDA 13 runtime
- Hugging Face `hf` CLI
- Approximately 180GB free for the 127.30GiB target, 2.18GiB draft, runtime image, and caches

This exact profile was tested on Linux 6.17, NVIDIA driver 590.48.01, and Docker 29.1.3. Those are evidence, not universal minimum versions.

## Quick start

### 1. Configure

```bash
git clone https://github.com/samuelcardillo/glm-5.3-flash-2x-rtx-pro-6000-blackwell.git
cd glm-5.3-flash-2x-rtx-pro-6000-blackwell
cp config/example.env .env
$EDITOR .env
```

Set absolute `MODEL_DIR`, `DRAFT_DIR`, and `CACHE_DIR` paths. Select exactly two physical RTX PRO 6000 indices. The API binds to `127.0.0.1` by default.

Review the [DFlash2 terms](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) and retained [CC BY-NC-ND 4.0 text](THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt). If appropriate for your use, set:

```text
ACCEPT_DFLASH2_RESEARCH_LICENSE=1
```

### 2. Download both immutable revisions

```bash
ACCEPT_DFLASH2_RESEARCH_LICENSE=1 scripts/download-model.sh \
  /absolute/path/to/GLM-5.3-Flash-EXL3-K3-v1 \
  /absolute/path/to/GLM-5.3-Flash-DFlash2
```

The downloader writes `RECIPE_PIN.txt` only after both pinned downloads complete.

### 3. Preflight

```bash
scripts/preflight.sh
```

Preflight validates immutable pins, 16 target shards and exact byte total, DFlash2 architecture/size, profile boundaries, runtime image, GPU class/memory, exact two-device selection, and P2P read access.

### 4. Launch

```bash
scripts/serve.sh
```

The launcher automatically derives a hash-pinned chat template into `CACHE_DIR`; it never mutates the model snapshot. Its source is Z.ai's official Flash template at immutable revision `a5b45eb41df6402735dedc900be14a42e8d5e538`, including the tool-result reordering early-exit fix. The derivation changes only the two thinking-control expressions so thinking-off requests produce clean final content while preserving explicit reasoning modes.

Startup intentionally performs extensive graph and kernel warmup. Do not treat `/health` alone as release readiness; wait for Docker health to become `healthy` or use `scripts/wait-ready.py`.

### 5. Verify

```bash
python3 scripts/verify.py --base-url http://127.0.0.1:8000 --model glm-5.3-flash-local
python3 scripts/verify-vision-limit.py --base-url http://127.0.0.1:8000 --model glm-5.3-flash-local --output vision.json
python3 scripts/verify-multi-needle.py --base-url http://127.0.0.1:8000 --model glm-5.3-flash-local --tokens 1000000 --max-tokens 512 --output million.json
```

The vision verifier sends 1, 4, and 16 generated numbered images and requires the exact ordered values, then requires image 17 to be rejected. The long-context verifier constructs exactly 1,000,000 server-tokenized prompt tokens and retrieves six records placed at 5%, 25%, 50%, 75%, 95%, and 99%.

### 6. Benchmark

```bash
python3 scripts/benchmark-dflash2.py \
  --base-url http://127.0.0.1:8000 --model glm-5.3-flash-local \
  --suite code-agent --dflash-tokens 5 --concurrency 1 2 4 8 16 \
  --output-tokens 256 --warmup-runs 2 --runs 5 --output decode.json

python3 scripts/benchmark-prefill-v06.py \
  --base-url http://127.0.0.1:8000/v1 --model glm-5.3-flash-local \
  --profile fp8 --prompt-tokens 8192 16384 32768 65536 128000 \
  --runs 3 --output prefill.json
```

## Service and canary operations

Install the included user service with:

```bash
scripts/install-user-service.sh
systemctl --user start glm53-2x-rtxpro6000.service
```

For disruptive candidate testing, `scripts/run-canary.sh` prevalidates profiles, traps `EXIT`, `INT`, `TERM`, and `HUP`, removes the candidate, restarts the original service, and verifies its exact alias/context. Restoration failure overrides the test status.

## API exposure

The endpoint is unauthenticated and loopback-only by default. For remote use, bind only to a specific trusted LAN/Tailnet address or place an authenticated gateway such as LiteLLM in front. Never expose it directly to the public Internet.

## Reproducibility and privacy

No weights, private environment files, raw responses, benchmark fixtures, hostnames, GPU UUIDs, PCI IDs, or private paths belong in the repository. Public evidence contains aggregate measurements and hashes only.

Run local validation with:

```bash
scripts/ci.sh
```

Original downstream scripts and documentation are Apache-2.0. Models, DFlash2, the runtime image, CUDA components, and dependencies retain their own licenses.