# Validation and benchmark notes

These numbers are acceptance evidence for one exact deployment, not universal performance claims.

## Boundaries

- Decode throughput excludes TTFT unless explicitly stated.
- TTFT includes server tokenization/prefill as seen by the HTTP client.
- Vision latency is end-to-end for one generated PNG and short answer.
- KV pool/concurrency are vLLM startup reports, not physical-memory arithmetic.
- Long-context runs used the same checkpoint, image, GPU pair, cap, and KV profile.

| Workload | Result |
|---|---:|
| Exact 128,000-token needle retrieval | pass |
| Exact 261,900-token needle retrieval | pass |
| 261,798-token prompt | 69.33 seconds TTFT |
| Vision semantic number test | `73`, 8.91 seconds over LAN |
| Tool call after vision cutover | correct structured arguments |
| Zcode attached image to `Write` | exact `73\n` file |

Text-only and vision are separate lineages:

| Profile | KV pool | 262K concurrency estimate |
|---|---:|---:|
| `--language-model-only` | 734,003 | 2.80× |
| Vision: 16 images, video off | 545,259 | 2.08× |

Vision loads about 1.05GiB of BF16 visual tensors and changes allocation.

Reproduce short acceptance with `python3 scripts/verify.py`. Reproduce exact 128,000- and 261,900-token needle retrieval with:

```bash
python3 scripts/verify-long-context.py \
  --base-url http://127.0.0.1:8000 \
  --model glm-5.3-flash-local \
  --targets 128000,261900
```

The long-context script constructs exact chat-prompt lengths through the server's `/tokenize` endpoint and verifies `usage.prompt_tokens`. A pre-publication run on the qualified service passed at 128,000 tokens in 29.635 seconds and 261,900 tokens in 57.372 seconds. These timings are a separate request lineage from the historical 261,798-token/69.33-second TTFT row above.

The later large-task/repetition investigation, including raw-stream context sweeps, the thinking-control fix, and the matched ReplaySSM red/green isolation (11/80 shared-prefix loops plus an EngineCore crash with ReplaySSM, versus 0/120 loops/errors without it), is documented in [2026-09-01-corruption-investigation.md](2026-09-01-corruption-investigation.md).

## Qualified runtime overlays (2026-09-01)

The accepted derived image is `local/glm53-runtime-fixes:780ae1d07a501f61f7a2b6cb829eaff123c9661236a3f62f4bedd909e8e56d70`. It contains the two official XGrammar correctness backports, DCP-aware sparse-indexer workspace sizing, and the mixed-prefill scheduler overlay. The qualified profile enables `MIXED_PREFILL_CHUNK=skip`; the overlay's default `off` mode preserves stock scheduling.

### Workspace result

Profiler-level A1/A2 comparison recovered 0.99 GiB of usable KV-cache memory per rank. Reported KV-cache capacity increased from 365,782 to 573,058 tokens, a gain of 207,276 tokens (56.67%). Matched prose and structured TTFT/decode changes remained inside the 3% acceptance gate. A post-readiness `nvidia-smi` snapshot is not a workspace-saving measurement because vLLM reinvests reclaimed workspace into a larger KV cache.

### Mixed-prefill result

Five matched runs used five distinct-prefix fixtures of exactly 160,025 prompt tokens. Decode throughput below is usage-based and excludes the first streamed token; SSE event frequency is not treated as token throughput.

| Policy | Isolated decode | Active decode during prefill | Decoder wall | Prefill TTFT | Combined makespan |
|---|---:|---:|---:|---:|---:|
| `off` | 188.98 tok/s | 23.90 tok/s | 42.97 s | 40.50 s | 42.97 s |
| `skip` | 189.04 tok/s | 188.02 tok/s | 5.60 s | 44.41 s | 44.58 s |

`skip` recovered 99.42% of the measured decode loss. Aggregate throughput decreased 3.61%, within the 10% gate, and every prefill completed with valid ordering. Positive caps 128 and 512 were screened separately and rejected.

### Rejected spin-wait experiment

The isolated 16 ms spin-wait candidate reduced active decode CPU by only 6.15% at the median and 4.59% at P90, far below the required 50%. It is not part of the accepted image chain.

### Final candidate acceptance

- Default thinking off, explicit low/max thinking, native and concurrent/multiple tools, strict JSON schema, one image, five distinct images, image cap 16, and video rejection: pass.
- Exact 32,768- and 261,900-token needle retrieval with server-reported prompt counts: pass.
- Fresh-process 261,900-token prefill: 78.516 s cold TTFT, zero cold cache hits; 4.831 s warm TTFT with a 92.86% block-aligned hit rate.
- Repetition regression: 80 requests (40 thinking off, 40 maximum thinking) at 32,755 prompt tokens and concurrency 4; zero loops and zero errors.
- Long generation: 2,304 completion tokens in 12.295 s, no bounded repetition finding, concurrent distinct-prefix request passed, engine healthy afterward.
- Five-run prose: median TTFT 0.1582 s and median decode 85.78 tok/s.
- Five-run structured output: median TTFT 0.1581 s and median decode 142.05 tok/s.
- Candidate teardown and restoration of the proven endpoint: pass.

These finite tests do not guarantee that arbitrary autoregressive requests can never repeat or starve. In particular, `skip` deliberately defers a long prefill while an already-running decode remains active; the measured trade-off must be considered for continuously saturated workloads.

Archive JSON with `nvidia-smi`, topology, `docker inspect`, and logs. Never compare results unless checkpoint revision, template hash, image digest, arguments, hardware, context, and requests match.
