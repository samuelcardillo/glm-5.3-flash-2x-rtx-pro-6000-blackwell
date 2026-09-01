# Frozen provenance

This recipe keeps the base model, quantized checkpoint, runtime, and local integration as separate lineages.

| Layer | Immutable artifact |
|---|---|
| Base model | [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16) at `f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| Quantized checkpoint | [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) at `5ab363a8dcf6405955fd5f99671e01a1c9fb124b` |
| Runtime source | [`tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx`](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx) at `3bff1d5fdbafcc3d9865abebddbfe1eef435adef` |
| Runtime image | `ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:da5cec95778bf6996660b52e28a6e51737fec69cfc3d508bf298c8a89f273ac5` |
| B12x fork in image | [`tpurtell/sparkinfer-glmrt`](https://github.com/tpurtell/sparkinfer-glmrt) at `988246c8b007c9c1c2006eb677f6fa4b26aeb561` |
| EXL3 vLLM fork in image | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| GLM base image | `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| EXL3 source image | `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| vLLM reported by image | `0.1.dev20051+g487ecf187` |
| A1 XGrammar recipe | `local/glm53-runtime-fixes:e91aebecd2907d9905c6f4520c30d49fa57f4272e9e738d46c0d3edccf3d35fc` |
| A2 workspace recipe | `local/glm53-runtime-fixes:4ef38e761892c69e7c8e90748dbc362dca405cd6cde756b4afafb68cc0babd39` |
| Accepted A3 scheduler recipe | `local/glm53-runtime-fixes:780ae1d07a501f61f7a2b6cb829eaff123c9661236a3f62f4bedd909e8e56d70` |
| Runtime stack reported upstream | Torch 2.13, CUDA 13, CUTLASS DSL 4.6.2 |

The checkpoint's current Hugging Face HEAD is newer. This recipe deliberately pins the exact revision tested with this runtime and profile.

The upstream runtime repository's original default checkpoint revision was `4739eb1bcfd478e8a32da6358908567bc3a9ac51` and carried the Z.ai MIT text. This recipe does **not** use that snapshot: `5ab363…` has a ShapleyMCG `LICENSE` whose directly downloaded SHA-256 is `9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a`. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The exact public source repository for EXL3/vLLM fork commit `300386…` and the complete source/build provenance for the `cstechdev` base image remain unresolved. This repository does not redistribute either image and does not represent the composed binary as fully source-audited.

## Derived runtime lineage

The three accepted local images are deterministic recipes, not redistributed container layers:

1. A1 layers only official vLLM XGrammar commits `12f64b39d29282437e35be9aa5db432fb2a1a6e6` and `c6e19b3be24338759a443e03c8325d76da9ee202` on the exact control image.
2. A2 layers only the DCP-aware sparse-indexer workspace overlay on exact A1. Its original `indexer.py` SHA-256 is `ef32db87f3735cf72adfac7cd8906a8951c69178e7eb1c0903965f2a1fca796c`; the accepted patched SHA-256 is `8d310b4782c3c9fbcf2d58e6f14236cbf3176eb0759289602a863754c1e5006c`.
3. A3 layers only the mixed-prefill scheduler overlay on exact A2. Its original scheduler SHA-256 is `a509e42c4b9b6ed388f1f5372a7b50d0aeab58f959113812f817f014febef103`; the patched SHA-256 is `4758a9f50e1234f872b017f3ce766f4493d87a1ce775c705dff87ee9988fb87e`.

The scheduler and spin-wait experiments were informed by inspection of `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks` at commit `c190db1ae17ba8dff20129ed1f308d10c63cf37d`, but its DGX-specific overlays were not copied unchanged. Local overlays are source-hash and anchor pinned, transactional, idempotent, and fail closed on unsupported source or configuration.

A separately isolated 16 ms spin-wait image was tested and rejected because it failed the required CPU-reduction gate. Only its aggregate result is documented; its rejected source and recipe are not part of this repository or the accepted A3 chain.

## Qualified host

- Linux `6.17.0-23-generic`
- NVIDIA driver `590.48.01`
- Docker `29.1.3`
- Physical GPU 0: RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887MiB
- Physical GPU 2: RTX PRO 6000 Blackwell Workstation Edition, 97,887MiB
- An unrelated RTX 5090 at physical GPU 1 was excluded with `GPU_DEVICES=0,2`
- CUDA P2P read status between selected GPUs: `OK`

These are validation facts, not universal minimum-version claims.

## Qualified launch profile

- TP2 and DCP2 (`ag_rs`)
- B12X MLA sparse attention
- NVFP4 MLA KV cache
- 262,144 maximum model length
- 2,048 maximum batched tokens
- 16 scheduler sequence slots
- 0.950 GPU memory utilization
- Prefix caching with aligned Mamba cache
- Feedback-adaptive MTP K1–K5
- Standard speculative full-state rollback; ReplaySSM disabled after a matched corruption/crash regression
- Maximum 16 images and 0 videos per prompt
- GLM 4.7 tool parser and GLM 4.5 reasoning parser

## Local acceptance evidence

| Check | Result |
|---|---:|
| Text-only KV pool before vision | 734,003 tokens |
| Vision-enabled KV pool | 545,259 tokens |
| Full 262,144-token concurrency estimate with vision | 2.08× |
| Exact 128,000-token needle retrieval | pass |
| Exact 261,900-token needle retrieval | pass |
| Recipe verifier retest, 128,000 / 261,900 exact chat tokens | pass, 29.635 / 57.372 seconds |
| 261,798-token prompt TTFT | 69.33 seconds |
| Structured tool call after vision cutover | pass |
| Semantic image test (`73`) | pass, 8.91 seconds over LAN after service-managed restart |
| Zcode image → model → `Write` | pass; exact output `73\n` |

These are local observations, not vendor claims. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Local integration changes

1. Bound context to 262,144 rather than using the runtime repository's larger experimental profile.
2. Selected physical GPUs explicitly on a mixed-GPU workstation.
3. Enabled multimodal input with a 16-image limit and video disabled.
4. Repaired the checkpoint's text-only chat template with checksummed GLM image placeholders.
5. Added user-systemd packaging and end-to-end checks.

No model weights or container layers are redistributed here.
