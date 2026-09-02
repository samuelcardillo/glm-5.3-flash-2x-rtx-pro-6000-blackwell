# Frozen provenance

This recipe treats the base model, quantized target, speculative drafter, runtime image, derived template, and downstream integration as separate lineages.

| Layer | Immutable identity |
|---|---|
| Base model | `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e` |
| Quantized target | `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| Runtime recipe | `tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx@b91f092861d4f2534a4cb8053a6552c714bacba7`, tag `v0.6.0` |
| Runtime source recorded by image | `d6fbe22…` |
| Runtime OCI index | `sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b` |
| Runtime amd64 manifest | `sha256:4858ebd…` |
| B12x fork in v0.6 image | `tpurtell/sparkinfer-glmrt@611ffe8…` |
| DFlash vLLM delta | `b389ac2…` |
| EXL3/vLLM fork lineage | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| vLLM reported by image | `0.1.dev20051+g487ecf187` |
| Base GLM image | `sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` |
| EXL3 source image | `sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` |
| Original K3 chat template | `sha256:34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891` |
| Derived thinking-control template | `sha256:5bcdf9be4e5b4a6cf2017f74f7e0b5c7f91bb814a275438dc678dd48da1f81b5` |

Abbreviated image-internal commit identities above are preserved as reported by the upstream v0.6 provenance. The runtime recipe commit and OCI digests are the portable acquisition anchors.

The `v0.6.0` tag is annotated but unsigned. This recipe pins the Git commit and OCI digest independently and does not treat the tag name as authenticated provenance.

## Checkpoint identity

The target contains 16 safetensors shards totaling exactly 136,686,260,192 bytes. Its model card describes K3 EXL3/MCG quantization of all 288 routed experts, 37,152 quantized projections, and 1,618 native tensors / 18.01GiB retained byte-for-byte from the base source. The target card declares MIT but the pinned snapshot has no standalone `LICENSE`; this recipe retains the base-model MIT text and discloses that boundary rather than inventing a target license file.

The DFlash2 draft contains a 2,342,169,800-byte `model.safetensors`, identifies architecture `DFlash2DraftModel`, and declares CC BY-NC-ND 4.0 for research/evaluation. Commercial licensing requires separate permission from Inco AI. It is downloaded independently and is not covered by this repository's Apache-2.0 license.

## Runtime delta from the superseded profile

The v0.6 image replaces the previous v0.3 plus local A1–A3 chain. Material additions include:

- DFlash2 V2 and GLM EAGLE3 target taps;
- independent replicated DFlash KV with 128-token allocation blocks;
- DFlash-aware prefix hashing and DCP1 draft / DCP2 target separation;
- EXL3 global-to-local EP2 expert loading;
- B12x EP MoE planning/execution and graph-stable MCG scratch;
- current DSA indexer integration;
- mHC and release-readiness warmup.

The former Brandon-specific template patch, XGrammar/workspace/mixed-prefill derived images, native adaptive MTP and ReplaySSM controls are not layered onto v0.6. Source-pinned overlays qualified against the old runtime cannot be transferred blindly.

## Qualified launch profile

- TP2 + EP2 + DCP2 (`ag_rs`)
- B12x sparse MLA attention
- FP8 DS MLA target KV cache
- DFlash2 K5 with replicated BF16 draft KV
- 1,048,576 maximum model length
- 2,048 maximum batched tokens
- 16 scheduler sequences
- 0.950 GPU memory utilization
- Prefix caching and aligned Mamba cache
- 16 images, zero videos
- GLM 4.7 tool parser and GLM 4.5 reasoning parser
- Thinking disabled by default through the derived template
- ReplaySSM absent
- Loopback publication and exact two-device selection

## Qualified host

- Linux `6.17.0-23-generic`
- NVIDIA driver `590.48.01`
- Docker `29.1.3`
- One RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887MiB, measured at 300 W
- One RTX PRO 6000 Blackwell Workstation Edition, 97,887MiB, measured at 600 W
- An RTX 5090 was excluded
- CUDA P2P read status between selected GPUs: `OK`

Hardware facts are intentionally stated by class without publishing UUIDs, PCI addresses or topology output.

## Local qualification

- Reported KV cache: 2,926,692 tokens; 2.79 maximum-context request equivalents
- Exact 1,000,000-token six-needle retrieval: 6/6, 284.859 seconds, clean stop
- Ordered 1, 4 and 16-image requests: pass; image 17 rejected
- Text, semantic image and structured tool checks: pass before and after the 1M request
- Strict decode medians C1/C2/C4/C8/C16: 179.50 / 317.07 / 493.40 / 759.72 / 1,045.47 tok/s
- Exact 8K–128K prefill medians: 4,107–4,259 prompt tok/s
- Repetition regression: 80 requests, zero loops and zero errors
- Seven-case semantic content suite: 7/7 pass; 39.30% aggregate DFlash acceptance
- Candidate teardown and restoration of `overlord-testing` at 262,144 tokens: pass

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for metric definitions and limitations.

## Supply-chain limits

The public source/build provenance for the EXL3 source-image fork and the complete `cstechdev` base image remains incomplete. The v0.6 composed image also contains separately licensed dependencies that are not fully described by a single OCI license label. This repository references but does not redistribute the image and does not represent it as fully source-audited.

No model weights, draft weights, container layers or private receipts are redistributed here.