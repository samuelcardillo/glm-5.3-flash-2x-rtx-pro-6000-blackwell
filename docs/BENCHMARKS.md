# Qualified v0.6 results

These are local observations for one immutable profile, not universal performance claims.

## Frozen profile

- Target: `wrldsuksgo2mars/GLM-5.3-Flash-EXL3-K3-v1@319d66a8b53092b491f698440ecea781e4ddd4e4`
- DFlash2: `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
- Runtime OCI index: `sha256:fe249b88d091430d8a88cd987d087d556053f0f067a649f2e9ca95895129e82b`
- TP2 + EP2 + DCP2; B12x sparse MLA
- FP8 target KV; replicated BF16 DFlash2 K5 draft KV
- Maximum context 1,048,576; batch cap 2,048; scheduler slots 16
- Maximum 16 images; video disabled
- Thinking off by default; ReplaySSM absent
- GPU power during measurements: Max-Q card 300 W, full workstation card 600 W

## Capability acceptance

| Check | Result |
|---|---:|
| Health, exact alias and 1,048,576 context | pass |
| Thinking-off exact `TEXT_OK` | pass |
| Structured tool call | pass |
| Semantic image number | pass |
| Ordered 1 / 4 / 16 images | pass |
| Image 17 rejection | pass |
| Exact 1,000,000-token six-needle retrieval | 6/6, 284.859 s |
| Post-1M functional and 16-image retest | pass |
| 32K/C4 repetition regression | 80/80, zero loops/errors |
| Seven-case semantic content suite | 7/7 pass; 39.30% DFlash acceptance |
| Candidate teardown and exact 262,144 control restoration | pass |

The six records were placed near 5%, 25%, 50%, 75%, 95%, and 99%. The live server reported exactly 1,000,000 prompt tokens. The successful retry used a 512-token output budget and completed with `finish_reason=stop`; the initial 128-token attempt retrieved too verbosely and ended at the output limit, so it was rejected rather than counted.

## Decode throughput

Method: fixed code-agent fixture, 256 forced output tokens per sequence, temperature 0.2, fixed seed, two warmups plus five measured runs per point. Each sequence is timed from its own first through last streamed token. Aggregate decode sums per-sequence rates and excludes TTFT. DFlash metrics are Prometheus counter deltas.

| Concurrency | Median | Min | Max | Median acceptance | Committed tokens / target pass |
|---:|---:|---:|---:|---:|---:|
| 1 | 179.50 | 176.89 | 190.35 | 64.92% | 4.246 |
| 2 | 317.07 | 293.49 | 358.56 | 65.45% | 4.273 |
| 4 | 493.40 | 472.10 | 497.96 | 65.97% | 4.298 |
| 8 | 759.72 | 715.26 | 773.97 | 66.32% | 4.316 |
| 16 | 1,045.47 | 993.33 | 1,065.17 | 66.46% | 4.323 |

Every measured point had positive draft and target-pass counters, `accepted + rejected = drafted`, exact usage/token counts, the expected finish reason, and an explicit SSE `[DONE]` marker.

### Upstream comparison

T.J. Purtell's final v0.6 evidence reports 222.6 tok/s at C1 and 1,067.2 tok/s at C16 with both GPUs explicitly capped at 400 W. The local C16 result is 2.0% lower. Upstream's initially selected profile reported 1,004.9 tok/s at C16, which the local result exceeds by 4.0%.

This host has an asymmetric GPU pair: the Max-Q card ran at 300 W and has a 325 W maximum; the full workstation card ran at 600 W. It cannot reproduce the upstream 400 W / 400 W condition. The lower C1 result must not be described as a runtime-only regression or speedup because hardware, power and historical checkpoint lineages differ.

## Prefill and TTFT

Method: exact-length unique prompts, no prefix reuse, three runs per point. TTFT is client request to first streamed token and includes server tokenization, prefill and one-token handoff.

| Prompt tokens | Median prompt tok/s | Min | Max | Median TTFT |
|---:|---:|---:|---:|---:|
| 8,192 | 4,258.74 | 4,251.83 | 4,264.21 | 1.924 s |
| 16,384 | 4,107.44 | 4,101.25 | 4,124.31 | 3.989 s |
| 32,768 | 4,219.75 | 4,218.10 | 4,221.59 | 7.765 s |
| 65,536 | 4,230.59 | 4,219.19 | 4,233.01 | 15.491 s |
| 128,000 | 4,199.66 | 4,196.44 | 4,202.44 | 30.479 s |

## Capacity

The runtime reported 2,926,692 KV-cache tokens and 2.79 maximum-context request equivalents. This is a scheduler report, not physical-memory arithmetic.

## Historical aligned profile

The superseded Brandon/TR3 4-bpw profile used a different checkpoint, NVFP4 KV, adaptive native MTP, local runtime overlays and a 262,144-token ceiling. Its key evidence was 573,058 KV tokens after workspace optimization, 189.04 tok/s isolated decode in a different fixture, exact 261,900-token retrieval, and 80/80 clean repetition requests. Those numbers are retained here only as historical lineage and are not a matched before/after benchmark.

## Evidence policy

Private receipts retain raw machine-bound evidence locally. Public documentation records aggregate timings, counters and immutable identities without prompts, responses, reasoning, paths, hostnames, GPU UUIDs, PCI IDs or logs. Decode throughput is never inferred from SSE event count.