# Large-task corruption investigation — 2026-09-01

## Scope

This investigation was triggered by a report that the aligned GLM-5.3 Flash EXL3 recipe worked for short tests but emitted random-looking words and repeated indefinitely on a larger work task. The reporter could not export the work trace, so all requests here are synthetic and contain no private data.

The tests used the recipe's pinned aligned checkpoint/runtime profile on two RTX PRO 6000 Blackwell GPUs. They do not prove identical behavior on AWS G7e hardware.

## Confirmed template/configuration fault

The checkpoint's original chat template did not inspect `enable_thinking`. It always:

1. selected maximum reasoning unless `reasoning_effort` was exactly `low` or `high`; and
2. opened every generation with `<think>`.

The integration profile advertised thinking as disabled, but its `enable_thinking=false` value had no effect on that template. A top-level OpenAI-compatible `enable_thinking` extension was also not a reliable template-argument transport in this runtime.

This is a real large-task failure mode. With temperature 0 and a 4,096-token output budget, three synthetic implementation/design tasks each ended with:

- `finish_reason=length`;
- 4,096 completion tokens;
- no final `content`; and
- the entire generation in `reasoning_content`.

Representative elapsed times were 48.79–55.97 seconds. A client that displays or concatenates `reasoning_content` can look stuck and can expose low-quality intermediate reasoning as apparently random output.

## Raw corruption observation

One 32,761-token synthetic retrieval request produced a genuine repetitive reasoning stream:

```text
We need answer to user. Needhandlehandlehandlehandle...
```

It exhausted all 512 completion tokens in 40.67 seconds and returned no final content. The raw SSE contained 505 consecutive `handle` deltas; tokenization maps unspaced `handle` to token ID 8191, so this was genuine server generation rather than client field assembly.

The initial small sample did not isolate a cause:

- replaying the identical request completed correctly;
- a serial context sweep passed at 8,198, 32,761, 65,531, 131,071, and 199,995 prompt tokens;
- all five sweeps recovered the expected synthetic needle;
- 20 controlled concurrent 32,761-token requests passed with the published speculative profile; and
- 20 more passed after speculative decoding and ReplaySSM were disabled.

The tighter matched stress regression did isolate the failing path. Every request used the same pinned checkpoint, runtime, GPUs, 32K fixture, temperature 0, seed, prefix-cache setting, adaptive MTP policy, concurrency 4, and 512-token bound. With ReplaySSM enabled:

- thinking off with a shared prefix produced 4/40 repetitive final-content streams;
- maximum thinking with a shared prefix produced 7/40 `handlehandle...` loops; and
- the unique-prefix phase completed four requests, then failed the remaining 36 after EngineCore died.

The fatal stack terminated in `materialize_kda_replayssm_state` with `ValueError: ReplaySSM prefill source/state row count mismatch`. The warning immediately before failure showed JIT compilation of `_reset_gdn_replayssm_spec_cursors_kernel`.

With **only** ReplaySSM disabled, adaptive MTP continued through standard full-state rollback. The corresponding 120 requests—40 thinking-off/shared-prefix, 40 maximum-thinking/shared-prefix, and 40 maximum-thinking/unique-prefix—completed with zero genuine loops, zero errors, and no engine death. One detector candidate was manually rejected as a normal stopped answer that mentioned the archive eight times while reasoning about the instruction.

This matched red/green result plus the direct ReplaySSM exception establishes the pinned runtime's ReplaySSM state path as the cause of the raw intermittent repetition/crash failure class. It does not implicate EXL3 weights, NVFP4 KV, prefix caching, adaptive MTP generally, context capacity, or AWS hardware.

## Controlled mitigation

The template patch now:

- honors `enable_thinking=false`;
- emits `<think></think>` when thinking is disabled;
- preserves `low`, `high`, and default maximum reasoning when thinking is enabled; and
- upgrades the earlier vision-only patch while preserving a checksum-verified original backup.

Independently, the safe runtime profile now forces `USE_REPLAYSSM=0`. Adaptive MTP remains enabled with baseline full-state rollback. Existing configurations that explicitly enable ReplaySSM fail preflight rather than silently selecting the known-bad path.

The launcher also supplies a server-side default:

```text
--default-chat-template-kwargs {"enable_thinking":false}
```

This protects clients that omit or mis-serialize the option. A request can opt back in with:

```json
{"chat_template_kwargs":{"enable_thinking":true,"reasoning_effort":"low"}}
```

After the fix, the same large design fixture with a 2,048-token budget produced 7,982 characters of final content, zero reasoning characters, and no repeated-subword flag in 25.02 seconds. Explicit low reasoning produced 8,341 final-content characters plus 71 reasoning characters in 24.15 seconds.

## Regression checks

After the template and launcher change:

- exact text: pass (`TEXT_OK`);
- native tool call: pass (`record_value({"value":"TOOL_OK"})`);
- genuine generated-image recognition: pass (`73`);
- one request containing five distinct generated images: pass (`red, green, blue, yellow, magenta`);
- exact 32,768-token needle retrieval: pass;
- exact 261,900-token near-limit needle retrieval after the template fix: pass in 73.939 seconds with the former profile;
- exact 261,900-token near-limit needle retrieval after disabling ReplaySSM: pass in 64.734 seconds;
- served context remains 262,144 tokens;
- adaptive MTP with standard full-state rollback and prefix caching remain enabled;
- ReplaySSM is disabled and rejected by configuration validation;
- a final persistent-service run of the public verifier completed 80/80 requests with zero loops and zero errors; and
- the minimum configured image limit is five, with the qualified default still sixteen.

## Interpretation

There were two independent faults:

1. The template ignored thinking-off, allowing hidden reasoning to consume the whole output budget.
2. ReplaySSM corrupted or mismatched recurrent speculative state under the matched concurrent 32K workload, producing real repeated-token streams and eventually killing EngineCore.

Both known recipe-level causes are removed from the qualified profile. No finite test can guarantee that an autoregressive model will never repeat under every adversarial prompt or malformed client feedback loop, so clients should still bound output and preserve reasoning/content separation. Repetition penalties are not the root fix for either fault.
