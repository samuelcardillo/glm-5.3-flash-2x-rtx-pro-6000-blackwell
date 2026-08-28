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

Archive JSON with `nvidia-smi`, topology, `docker inspect`, and logs. Never compare results unless checkpoint revision, template hash, image digest, arguments, hardware, context, and requests match.
