# Third-party notices and supply-chain disclosures

This repository publishes documentation and integration scripts only. It does not redistribute model weights or container layers. Users download those artifacts from their original registries and remain responsible for their terms. This ledger distinguishes verified source/license provenance from unresolved binary-image provenance.

## Model lineages

### Z.ai GLM-5.3 Flash

- Creator/publisher: **Z.AI Co., Ltd / Z.ai (`zai-org`)**
- Tested base: `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e`
- License at that revision: MIT, copyright 2026 Z.AI Co., Ltd
- Exact retained text: [`THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt`](THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt)

### Brandon Music / ShapleyMCG TR3 checkpoint used here

- Quantizer/publisher: **Brandon M. Music (`brandonmusic`)**
- Tested artifact: `brandonmusic/GLM-5.3-Flash-tr3-4bpw@5ab363a8dcf6405955fd5f99671e01a1c9fb124b`
- License at this exact revision: ShapleyMCG License 1.0. Direct download SHA-256 of `LICENSE`: `9a354667162e40201fa556e29ae7a327cdb112eacaa8ef100106e6063635e28a`.
- Exact retained text: [`THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt`](THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt)
- Required notice and citation are reproduced in README and ATTRIBUTIONS.

The upstream T.J. Purtell runtime repository originally pinned an older checkpoint snapshot, `4739eb1bcfd478e8a32da6358908567bc3a9ac51`, whose `LICENSE` was the Z.ai MIT text. **That is not the checkpoint revision deployed or downloaded by this recipe.** Do not transfer the older snapshot's MIT classification to `5ab363…` or current HEAD.

## Runtime and code lineages

### T.J. Purtell dual-GPU runtime

- Author/integrator: **T.J. Purtell (`tpurtell`)** and contributors
- Source: [`tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx`](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx) at `3bff1d5fdbafcc3d9865abebddbfe1eef435adef`
- License: Apache-2.0
- Contribution used here: container composition, GLM/EXL3/B12x integration, adaptive MTP, ReplaySSM GLM ports, launch profile, tests, and benchmark foundations.
- Modified-file notice: this repository's `scripts/serve.sh`, profile defaults, systemd packaging, template repair, and verification tooling are downstream integration adaptations and are not claimed to be upstream originals.

The source repository name contains `ext3`; the quantization technology is EXL3/TR3.

### B12x / SparkInfer

- Upstream author named in package metadata: **Luke Alonso**
- Upstream organization/contributors: **Local Inference Lab and B12x contributors**
- Upstream: [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x)
- GLM/EXL3/PCIe fork: **T.J. Purtell and fork contributors**, [`tpurtell/sparkinfer-glmrt`](https://github.com/tpurtell/sparkinfer-glmrt) at `988246c8b007c9c1c2006eb677f6fa4b26aeb561`
- License: Apache-2.0

### ExLlamaV3 / EXL3 / Trellis

- Primary author: **Turboderp**
- Contributors: **ExLlamaV3 contributors**
- Source: [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3)
- License: MIT, copyright 2025 Turboderp
- Contribution: EXL3/Trellis format, codebooks, dequantization, GEMM/PTX, and inference foundations vendored or adapted by the B12x/runtime lineage.

### vLLM and specifically credited contributors

- Project: **vLLM project and contributors**
- Source: [`vllm-project/vllm`](https://github.com/vllm-project/vllm)
- Base version lineage: commit [`487ecf187d3dfe74d2cf6119a92881dba403c219`](https://github.com/vllm-project/vllm/commit/487ecf187d3dfe74d2cf6119a92881dba403c219)
- License: Apache-2.0
- **Johnny-Liou**: ReplaySSM/speculative-decode PRs [#48792](https://github.com/vllm-project/vllm/pull/48792), [#49847](https://github.com/vllm-project/vllm/pull/49847), and [#49887](https://github.com/vllm-project/vllm/pull/49887).
- **CZT0**: dynamic speculative-decode graph fix [#49652](https://github.com/vllm-project/vllm/pull/49652).
- **ZJY0516**: upstream GLM-5.3 vLLM support reference [#53906](https://github.com/vllm-project/vllm/pull/53906).
- **Jared**: separately thanked by the pinned runtime author for GLM upstream work; no surname or account was supplied in that source, so no further identity is asserted here.

### Other named projects and organizations

- **MiaAI-Lab contributors**: nearby dual-DGX-Spark SM12x deployment and stability reference, [`MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark), MIT, copyright 2026 MiaAI-Lab.
- **Hugging Face team and Transformers contributors**: model configuration, processing, tokenization, and Hub distribution; Transformers is Apache-2.0.
- **PyTorch contributors**: tensor/compiler runtime. PyTorch uses its project license plus bundled third-party notices; preserve the complete installed notice bundle when redistributing an image.
- **NVIDIA Corporation and CUTLASS contributors**: CUDA, drivers, RTX PRO hardware, CUTLASS, CuTe DSL, and NVIDIA Container Toolkit. CUTLASS main source is BSD-3-Clause; CuTe DSL and CUDA components include NVIDIA EULA terms. See [CUDA EULA](https://docs.nvidia.com/cuda/eula/index.html) and [CuTe DSL license](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/license.html).
- **Docker/Moby, Docker CLI, and Compose contributors**: external container tooling, primarily Apache-2.0 source projects; Docker Desktop and hosted-service terms are separate.
- **`cstechdev`**: publisher of the pinned day-zero CUDA 13/SM120 GLM/vLLM base image.

## Unresolved binary-image provenance

These gaps do not cause this documentation repository to redistribute the affected binary code, but they are material supply-chain and downstream-redistribution concerns:

1. `ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:86c8c1054f9c24454949e37031ce6165c007963aa0c0ef30fa884f6d4170af32` supplies copied EXL3/vLLM adapter files. The corresponding public source repository for claimed commit `30038602b71395f481ef4a6edfe4fcf8551d9c15` was not located.
2. `cstechdev/vllm:glm53-flash-nope-sm120-cu130-20260826-r1@sha256:0bd709e80b8ff13ae5de8f7d7f708a499fade3a26970d56afb1be2ff3860fde5` does not expose a discoverable public patch/source repository or complete build revision in its labels.

Do not redistribute, mirror, or represent the composed runtime image as fully source-audited until those publishers provide source/build provenance and complete third-party notices. Apache-2.0 provenance for upstream vLLM does not by itself establish the licensing status of every modified binary or copied adapter file.

Corrections with primary-source evidence are welcome through GitHub issues.
