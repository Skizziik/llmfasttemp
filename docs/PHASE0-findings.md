# Phase 0 — de-risking findings

Per the TMS plan ([docs/TMS-research.txt](TMS-research.txt) §"Recommendations"),
Phase 0 instruments and de-risks *before* building anything. Status below.

## 0a. Byte-accurate roofline (DONE) — `bench/roofline.py`

Tensor byte sizes read directly from `gemma-4-E4B-it-Q4_K_M.gguf`:

| group | GiB | share |
|---|---:|---:|
| input_embedding (262144×2560, hi-precision) | 2.666 | 53.8% |
| transformer_blocks | 2.102 | 42.4% |
| output_proj | 0.135 | 2.7% |
| ple/altup | 0.051 | 1.0% |
| **total** | **4.954** | |

**Weight bytes read per decode token: ~2.288 GiB** (blocks + ple + output_proj;
the 2.67 GiB embedding table is an input lookup, *not* streamed per token — if it
were, the ceiling would fall below the measured speed, which is impossible, so
this confirms the accounting).

Memory-bound ceilings vs. our measured baseline (62.12 tok/s):

| | tok/s |
|---|---:|
| @ 272 GB/s (RTX 4060 raw BW) | 110.7 |
| @ 453 GB/s (NVIDIA L2-effective claim) | 184.4 |
| **measured baseline** | **62.1** |
| → efficiency vs raw ceiling (η) | **56%** |
| → implied effective BW used | 153 GB/s |

**Reading.** llama.cpp already runs at ~56% of the raw memory wall (η≈0.56, in
the doc's expected 0.5–0.7 band). There is some pure-efficiency headroom to
~110 tok/s, but it is bounded and hard-won. The only *large* lever is reducing
bytes-read *per emitted token* — i.e. emitting multiple verified tokens per
weight-read. **This quantitatively confirms the TMS thesis: speculative decoding
is the only path to a big speedup; kernel micro-optimisation is not.**

## 0b. MTP drafter availability (INVESTIGATED) — partial blocker

- ✅ A Gemma 4 E4B MTP/"assistant" drafter exists as GGUF:
  [`AtomicChat/gemma-4-E4B-it-assistant-GGUF`](https://huggingface.co/AtomicChat/gemma-4-E4B-it-assistant-GGUF)
  (F16/Q8_0/Q5_K_M/Q4_K_M/Q4_K_S).
- ❌ **Mainline `ggml-org/llama.cpp` (our build b9672) does not support it yet.**
  The head is `Gemma4AssistantForCausalLM` and needs a custom `--mtp-head` path.
  Tracked in [llama.cpp Discussion #22735](https://github.com/ggml-org/llama.cpp/discussions/22735).
- ✅ Support exists in forks: `ik_llama.cpp` (PR #1744, merged 2026-05-10,
  **2.6–2.98× lossless verified**) and `reffdev/llama.cpp` (gemma4-mtp branch);
  repro harness at [karany97/llamacpp-gemma4-mtp](https://github.com/karany97/llamacpp-gemma4-mtp).

**Consequence.** Reproducing MTP-only ~3× (the TMS "safe floor") requires building
or obtaining a fork with `--mtp-head`. A CUDA build on Windows is the main cost.
This is the next gating task before any TMS component can be measured.

## 0c. Nested-shell `c` measurement (BLOCKED, deferred)

Measuring the E2B shell's true draft-cost ratio `c` (TMS kill-criterion: must be
≤0.55) needs MatFormer shell extraction in a fused kernel — not exposed by
llama.cpp. Deferred until we have a custom runner or the fork.

## Next step

Build/obtain an MTP-capable llama.cpp fork → reproduce MTP-only speedup on this
exact GGUF + the assistant drafter → record it as the first real speedup row in
[BENCHMARKS.md](../BENCHMARKS.md). That is TMS Phase 1 (the lossless floor).
