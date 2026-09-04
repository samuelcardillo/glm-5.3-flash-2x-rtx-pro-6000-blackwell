# Third-party notices and supply-chain disclosures

This repository publishes integration scripts and documentation only. It does not redistribute model weights or container layers. Users acquire those artifacts from their original publishers and remain responsible for their terms.

## Z.ai GLM-5.3 Flash

- Creator/publisher: Z.AI Co., Ltd / Z.ai (`zai-org`)
- Base lineage: `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e`
- Updated official chat template: `zai-org/GLM-5.3-Flash-BF16@a5b45eb41df6402735dedc900be14a42e8d5e538`
- Vendored template SHA-256: `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`
- License at that revision: MIT
- Retained text: [`THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt`](THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt)

## K3 EXL3 target

- Publisher: `wrldsuksgo2mars`
- Artifact: `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4`
- Card declaration: MIT
- The pinned snapshot has no standalone `LICENSE` file. The base-model MIT text is retained, but this recipe does not fabricate or imply a separately retrieved target-license file.
- Target-card lineage: GPTQModel `0565af7ce20a93df9bbc0e5563d7c6f60916f41a`, EXL3 MCG K3, 16 shards, 127.30GiB.

## DFlash2 draft

- Publisher: Inco AI
- Artifact: `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
- License: CC BY-NC-ND 4.0 for research and evaluation
- Commercial use: requires separate permission from Inco AI
- Retained legal text: [`THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt`](THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt), SHA-256 `cb6303892198afb24723a78e59be37222ffd7494690fca00d9307347df50e0b6`

The draft is not bundled or modified. Its restrictions are not superseded by this repository's Apache-2.0 license or the runtime's license.

## T.J. Purtell v0.6 runtime

- Author/integrator: T.J. Purtell (`tpurtell`) and contributors
- Source: [`tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx`](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx) at `b91f092861d4f2534a4cb8053a6552c714bacba7`
- Tag: `v0.6.0` (annotated but unsigned)
- License: Apache-2.0 for the repository source
- Referenced image: OCI index `sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b`
- Contribution: GLM/DFlash2 V2 integration, EAGLE3 taps, independent draft KV, DCP-aware prefix hashing, DCP1 draft/DCP2 target separation, EXL3 EP2 loading, B12x MoE/MCG paths, sparse attention and release warmup.

The downstream launcher deliberately changes upstream operational defaults: it selects exactly two GPUs, restricts publication to a configured private/loopback address, pins the image digest, disables ReplaySSM, defaults thinking off, validates configuration as data, and delegates lifecycle ownership to systemd or the foreground process. Its derived template modifies the official template only to make thinking opt-out explicit.

## Other runtime lineages

- B12x/SparkInfer: Luke Alonso, Local Inference Lab, contributors, and T.J. Purtell's GLM fork; Apache-2.0 source lineage.
- ExLlamaV3/EXL3/Trellis: Turboderp and contributors; MIT source lineage.
- vLLM: vLLM project and contributors; Apache-2.0 source lineage.
- GPTQModel: GPTQModel project and contributors.
- Hugging Face/Transformers, PyTorch, NVIDIA, CUTLASS/CuTe, CUDA, NVIDIA Container Toolkit, Docker/Moby and their contributors retain their respective licenses and notices.

## Unresolved binary-image provenance

The composed v0.6 runtime copies or depends on code and binaries from lower images whose complete public source/build provenance and notice inventory are not available here:

1. The EXL3 source-image lineage identifies fork commit `30038602b71395f481ef4a6edfe4fcf8551d9c15`, but its exact public source repository remains unresolved.
2. The `cstechdev` CUDA 13/SM120 GLM/vLLM base image does not expose complete public build/source provenance.
3. A single Apache-2.0 OCI label does not establish the licensing status of every copied adapter, binary, NVIDIA component or model.

This repository references but does not mirror the image. Do not redistribute or represent the composed binary as fully source-audited until the publishers provide complete source/build provenance and third-party notices.

Corrections supported by primary-source evidence are welcome through GitHub issues.