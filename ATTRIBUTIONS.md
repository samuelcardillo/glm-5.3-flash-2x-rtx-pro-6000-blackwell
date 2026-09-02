# Attribution

This repository is an integration recipe. It does not claim authorship of GLM-5.3, the K3 checkpoint, DFlash2, EXL3, vLLM, B12x, CUDA, Docker, or the NVIDIA container stack.

## Models

- **Z.ai / `zai-org`** created and published GLM-5.3 Flash. The pinned base lineage is `zai-org/GLM-5.3-Flash-BF16@f12e0fe1f6b2ea274c11a569582edfd99d993c5e`, MIT. Its license text is retained in [`THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt`](THIRD_PARTY_LICENSES/ZAI-GLM-5.3-Flash-BF16-MIT.txt).
- **`wrldsuksgo2mars`** published the K3 EXL3/MCG target used here: `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4`. The card credits Z.ai, Brandon for earlier K4 qualification work, MiaAI-Lab, GPTQModel, ExLlamaV3, vLLM and B12x contributors.
- **GPTQModel contributors** supplied the quantization framework identified by the target card.
- **Turboderp and ExLlamaV3 contributors** created the EXL3/Trellis format and optimized inference foundations.

## DFlash2

- **Inco AI** published `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` and the DFlash2 work used for speculative decoding.
- The draft is licensed CC BY-NC-ND 4.0 for research/evaluation; commercial licensing requires separate permission from Inco AI. The legal text is retained in [`THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt`](THIRD_PARTY_LICENSES/CC-BY-NC-ND-4.0.txt).
- The model card requests citation of Inco AI's “DFlash 2: Keep Drafting Parallel” and the original ICML 2026 DFlash paper by Jian Chen, Yesheng Liang, and Zhijian Liu.

## Runtime and inference

- **T.J. Purtell (`tpurtell`) and contributors** built, integrated and qualified the dual-RTX-PRO-6000 v0.6 runtime: [`tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx`](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx), Apache-2.0. The runtime provides the DFlash2/GLM target taps, B12x paths, EXL3 EP2 loading, DCP2, compact caches, PCIe collectives, warmup and benchmark foundations adapted here.
- **Luke Alonso, Local Inference Lab, B12x/SparkInfer contributors, T.J. Purtell and fork contributors** provided sparse MLA/indexer, SM12x kernels, MoE, MCG and collective work. Upstream: [`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x); GLM fork: [`tpurtell/sparkinfer-glmrt`](https://github.com/tpurtell/sparkinfer-glmrt).
- **vLLM contributors** provided serving, scheduling, cache management, multimodal processing, parsers and speculative-decoding foundations.
- **ZJY0516** authored the upstream GLM-5.3 vLLM support reference in PR #53906.
- **CZT0** authored the dynamic speculative-decode graph fix in PR #49652.
- **MiaAI-Lab contributors** published nearby dual-GPU/DGX Spark references acknowledged by the target and runtime lineages.
- **cstechdev** published the CUDA 13/SM120 GLM vLLM base image used in the runtime lineage.

## Ecosystem

Hugging Face, Transformers, PyTorch, NVIDIA, CUTLASS/CuTe, CUDA, NVIDIA Container Toolkit, Docker/Moby, and their contributors provide artifact distribution, model processing, tensor/compiler execution, kernels, drivers and container infrastructure under their respective licenses.

## Downstream contribution

This repository contributes a safety-hardened integration profile: immutable artifact pins, strict non-executable configuration parsing, loopback publication, exact GPU selection, a derived thinking-control template, research-license gating, canary restoration, privacy-safe capability tests, and local 1M/16-image/performance evidence.

Modified downstream files are not represented as upstream originals. Corrections with primary-source evidence are welcome through GitHub issues.