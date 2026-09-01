# Attribution and third-party notices

This repository is an integration recipe. It does **not** claim authorship of GLM-5.3, ShapleyMCG/TR3, EXL3, vLLM, B12x/SparkInfer, ReplaySSM, CUDA, Docker, or the NVIDIA container stack.

## Required ShapleyMCG attribution

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```

The tested [`brandonmusic/GLM-5.3-Flash-tr3-4bpw`](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw) checkpoint was created and published by **Brandon M. Music** using ShapleyMCG/TR3. It is source-available under the ShapleyMCG License 1.0, not OSI open source. A verbatim tested copy is retained at [`THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt`](THIRD_PARTY_LICENSES/ShapleyMCG-LICENSE-1.0.txt).

## Runtime and inference work

- **T.J. Purtell (`tpurtell`)** — built and published the dual-RTX-PRO-6000 runtime repository and pinned image; integrated and qualified EXL3 loading, B12x paths, DCP2, compact KV layouts, PCIe collectives, ReplaySSM, and adaptive MTP. [`tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx`](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx), Apache-2.0.
- **Luke Alonso, Local Inference Lab, and B12x/SparkInfer contributors**, plus **T.J. Purtell and fork contributors** — sparse MLA/indexer, SM12x kernels, compact cache paths, and collectives. Upstream [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x); GLM fork [`tpurtell/sparkinfer-glmrt`](https://github.com/tpurtell/sparkinfer-glmrt), Apache-2.0.
- **vLLM project contributors** — OpenAI-compatible serving, scheduling, multimodal processing, cache management, parsers, and speculative-decoding foundations. [`vllm-project/vllm`](https://github.com/vllm-project/vllm), Apache-2.0.
- **Johnny-Liou** — ReplaySSM/speculative-decode vLLM PRs [#48792](https://github.com/vllm-project/vllm/pull/48792), [#49847](https://github.com/vllm-project/vllm/pull/49847), and [#49887](https://github.com/vllm-project/vllm/pull/49887), ported and extended for GLM by the pinned runtime.
- **CZT0** — dynamic speculative-decode graph fix [#49652](https://github.com/vllm-project/vllm/pull/49652).
- **ZJY0516** — upstream GLM-5.3 vLLM support reference [#53906](https://github.com/vllm-project/vllm/pull/53906).
- **Jared** — thanked by the pinned runtime author for GLM upstream work. The upstream source provides no verifiable surname or handle, so this recipe preserves the credit without guessing an identity.
- **ExLlamaV3 contributors, including turboderp** — EXL3 format, trellis quantization, and optimized local-inference foundations. [`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3), MIT.
- **cstechdev** — published the pinned CUDA 13/SM120 GLM vLLM base image used by the runtime.
- **MiaAI-Lab contributors** — nearby SM12x stability work for dual DGX Spark systems and scheduler/spin-wait experiments inspected as comparative prior art; workstation overlays were independently reimplemented and validated rather than copied unchanged. [`MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark) and [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks).

## Model and ecosystem

- **Z.ai / `zai-org`** — created and published GLM-5.3 Flash. Tested base lineage: [`zai-org/GLM-5.3-Flash-BF16`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16), MIT at the pinned revision. Its exact license text is retained at [`THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt`](THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt).
- **Hugging Face Transformers and Hub contributors** — tokenizer, processor, packaging, and artifact distribution. Transformers is Apache-2.0.
- **NVIDIA** — CUDA, drivers, container integration, NVIDIA Container Toolkit, and RTX PRO 6000 Blackwell. NVIDIA Container Toolkit is Apache-2.0; CUDA/drivers retain NVIDIA terms.
- **Docker/Moby contributors** — container execution and distribution components under their respective licenses.
- **PyTorch contributors** — tensor/compiler runtime and its bundled third-party components.
- **NVIDIA CUTLASS/CuTe contributors** — SM12x kernel foundations; CUTLASS and CuTe/CUDA components retain their respective BSD and NVIDIA EULA terms.

The exact checkpoint and base-model license texts, named-contributor ledger, and unresolved binary-image source/build gaps are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## This repository's contribution

The original contribution here is deployment integration and documentation: a conservative 262K profile, physical-GPU selection, reversible multimodal template repair, user-systemd packaging, Zcode configuration, acceptance scripts, and a local measurement record. It builds on—and is subordinate to—the upstream work and licenses above.

If attribution is incomplete or inaccurate, open an issue with a primary-source link so it can be corrected promptly.
