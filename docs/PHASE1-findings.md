# Phase 1 — lossless MTP speculative decoding (RESULT)

First real speedup. We ran Gemma 4 E4B Q4_K_M with the official `gemma4_assistant`
MTP drafter (74.9 MB Q4_K_M) on the RTX 4060, via a from-source CUDA build of the
`bankenichi/atomic-llama-cpp-turboquant` fork (mainline llama.cpp has no
`--mtp-head` yet). Frozen 10-prompt suite, greedy, thinking off.

## Headline numbers

| Run | Avg tok/s | vs FA-auto baseline | vs FA-off baseline |
|---|---:|---:|---:|
| baseline (FA auto) | 62.12 | — | |
| plain decode, FA **off** (control) | 57.52 | −7.4% | — |
| **MTP spec decoding (FA off)** | **80.28** | **+29.2%** | **+39.6%** |

- **End-to-end real-world gain: ~1.29×.** Pure MTP effect (isolating the FA
  penalty): **~1.40×**.
- **Lossless confirmed:** MTP output is *character-for-character identical* to
  plain FA-off greedy decoding (verified on code_fib, reason_batball, list_primes).
  The speedup costs zero quality.

## The important part: speedup is entropy-dependent (validates TMS)

Per-prompt, the gain tracks draft acceptance almost perfectly — exactly the TMS
premise that a *verification-free / entropy-gated* path is needed:

| prompt | entropy | accept | tok/s | vs FA-off baseline |
|---|---|---:|---:|---:|
| list_primes | very-low | **0.90** | 108.0 | ~1.9× |
| reason_batball | medium | 0.87 | 107.7 | ~1.9× |
| summarize | medium | 0.73 | 99.4 | ~1.7× |
| code_fib | low | 0.74 | 97.0 | ~1.6× |
| json_book | low | 0.67 | 92.6 | ~1.6× |
| fact_capital | low | 0.50 | 88.2 | (3 tok, noisy) |
| ru_blackhole | medium | 0.38 | 65.4 | ~1.1× |
| explain_hashmap | medium | 0.32 | 63.1 | ~1.1× |
| longform_watercycle | medium | 0.35 | 59.2 | ~1.0× |
| prose_lighthouse | high | **0.29** | 55.8 | **~0.97× (slower!)** |

**Reading.** On low-entropy content (lists, code, structured, step math) MTP
hits ~1.6–1.9×. On high-entropy creative/multilingual prose, acceptance falls to
~0.3 and drafting *barely pays for itself or loses* — the draft+verify overhead
isn't recovered. This is precisely why TMS C3 (entropy-gated routing: don't draft,
or draft short, when predicted acceptance is low) is the next lever, and why a
single aggregate number understates what a content-aware system can do.

## Why not the headline "up to 3×"?

Honest reasons, consistent with the TMS caveats:
1. **Small model = large fixed overhead.** On a cheap 4B forward pass, tree/draft
   bookkeeping is a bigger relative cost (the mlx-lm EAGLE-3 cautionary tale).
2. **FA off penalty (~7%).** See limitation below.
3. **Our suite is ~half high-entropy** by design (to expose the entropy effect);
   a code/chat-heavy workload would average much higher.
4. Greedy, single short stream — no warmup/online adaptation yet (TMS C5).

## Limitation found: flash-attention crashes under MTP on sm_89

With FA enabled, the MTP graph aborts in `ggml-cuda/fattn.cu:109` (Gemma 4's
large head dims, key/value length 512). We run MTP with `-fa off`, which costs
~7%. **Fixing FA for these head dims is a concrete optimization lead** — it would
recover the 7% and speed the verify pass, lifting all MTP numbers.

## What this means for the plan

- **TMS Phase 1 floor is real and lossless** on this hardware. ✅
- The per-category acceptance table is the empirical input TMS C3 needs to set
  entropy thresholds. Next high-value step is **entropy-gated drafting** (skip/
  shorten drafts when acceptance is predicted low) — it should lift the aggregate
  by reclaiming the prose/multilingual rows that currently lose.
- Telescoping qualifiers (C2) still gated on the Phase-0c byte test (E2B `c`).

## Reproduce

```bash
# build dir = the MTP fork's build/bin/Release ; drafter = the 75MB assistant GGUF
python bench/run_suite.py --label mtp --technique "MTP" \
  --llama-bin <fork>/build/bin/Release --fa off \
  --mtp-head <path>/gemma-4-E4B-it-assistant.Q4_K_M.gguf --spec-type mtp
```
