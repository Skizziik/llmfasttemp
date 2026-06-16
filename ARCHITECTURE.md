# Presto — Architecture & Design

> North-star document. Honest about what's a known technique vs. our own bet,
> and honest about expected speedups. No magic numbers.

## 0. The one law everything obeys

Single-user, single-stream decoding is **memory-bandwidth bound**, not
compute-bound. To produce one token, you stream *every weight* of the model from
memory through the ALUs once. On an 8 GB consumer GPU running a model that fits
in VRAM, you are limited by how fast you can read those weights.

Consequence: you cannot meaningfully "out-kernel" llama.cpp on the same model —
its kernels already saturate bandwidth. The only large lever is **algorithmic**:
emit more than one token per weight-read. That is speculative decoding, and it's
the entire reason Presto exists.

```
tokens/sec  ≈  (bandwidth / model_bytes)  ×  avg_accepted_tokens_per_pass
              └──── fixed by HW + quant ────┘   └──── what WE optimize ────┘
```

## 1. Target model: Gemma 4 E4B (and why it's the right pick)

- **MatFormer (Matryoshka) architecture**: E2B is a *strict subset* of E4B's
  weights via FFN nesting. A smaller, fully-functional model lives inside the
  bigger one.
- **Per-Layer Embeddings (PLE)**: layer-specific embeddings live outside the hot
  weight memory and are paged in as needed — shrinks the resident footprint.
- **Effective ~4B params** → fits 8 GB VRAM at Q4/Q5 **with no CPU offload**.
  No PCIe round-trips. This is why it can feel instant where 12B cannot.

## 2. The speed stack (outermost = our novel bet)

### 2.1 ⭐ Matryoshka self-speculation  *(our research bet)*

Standard speculative decoding needs a *separate* draft model: extra VRAM, a
separate vocab/tokenizer to keep aligned, and often separate training. MatFormer
removes all three problems at once:

- **Drafter = E2B**, which is already resident as a subset of E4B's weights.
- **Verifier = E4B**, the full model.
- They share weights, vocabulary, and training → drafts are "the same model,
  just coarser" → expected high acceptance length.
- **Zero extra VRAM for the drafter.** Critical on 8 GB.

Honesty: self-speculative decoding exists in the literature (e.g. LayerSkip,
self-drafting). Exploiting *MatFormer nesting specifically* as the draft path is
the part that's underexplored — this is our empirical bet, not guaranteed. We
measure acceptance length and tok/s vs. baseline and let the numbers decide.

### 2.2 Adaptive draft length (entropy-gated)

Draft long when the model is confident (boilerplate, code, lists — low entropy),
short when it's uncertain. Avoids wasting verify passes on doomed long drafts.
Gate on the verifier's running entropy / the drafter's own confidence.

### 2.3 Hybrid drafter pool — add n-gram / prompt-lookup

For repetitive or structured output (code, JSON, quoted context), a cheap
prompt-lookup n-gram drafter gets free, instant hits with no model call. Run it
alongside the model drafter and take whichever proposes a longer accepted run.

### 2.4 Tree verification (EAGLE-2 style)

Instead of one linear draft, build a small *tree* of candidate continuations and
verify them all in **one batched forward pass** of E4B. Raises accepted tokens
per pass beyond a single linear guess.

### 2.5 Tight decode loop — CUDA graphs

For a ~4B model, per-token kernel-launch overhead is a real fraction of the
budget. Capture the decode step as a CUDA graph and replay it — this is a big
part of what turns "fast" into "instant" for small models.

### 2.6 Foundation (gets us to baseline, not past it)

Q4/Q5 weight quantization · quantized KV-cache (Q8) · fused attention ·
cuBLAS for GEMM (we do **not** rewrite matmul). These just reach the llama.cpp
baseline; the speedup over baseline comes from 2.1–2.5.

## 3. Honest speedup expectations

| Scenario | Realistic tok/s feel | vs. baseline |
|---|---|---|
| E4B baseline (autoregressive, in-VRAM) | fast already | 1.0× |
| + Matryoshka self-spec + tree verify | "instant" | ~1.8–2.5× |
| + CUDA graphs + adaptive + n-gram | "snappy as a 1B" | stack on top |

"20× vs llama.cpp on the same model" is not physical. "Feels instant / like a
tiny model" is achievable — and that's the actual user goal.

## 4. Backend abstraction

Everything plugs into one interface (`presto/backends/base.py`):

```python
class Backend:
    name: str
    def generate(self, prompt: str, max_tokens: int, **opts) -> Iterator[Token]
```

A `Token` carries text plus optional telemetry (e.g. how many draft tokens were
accepted this step) so the playground can *visualize* speculation live. We swap
backends — mock → reference → Presto — without touching the server or UI, and
benchmark them against each other in the same chat window.

## 5. Phase plan — TMS (Telescoping Matryoshka Speculation)

Plan adopted from [docs/TMS-research.txt](docs/TMS-research.txt); supersedes the
earlier E2B-as-drafter plan (see the banner in [RESEARCH.md](RESEARCH.md)).

- **Phase 0 — playground + de-risk** *(done)*: stdlib server + browser chat +
  live tok/s on a mock backend; honest llama.cpp baseline **62.12 tok/s**;
  byte-roofline (η≈0.56, ceiling ~110 tok/s — see
  [docs/PHASE0-findings.md](docs/PHASE0-findings.md)). Confirms speculative
  decoding is the only big lever.
- **Phase 1 — lossless MTP core** *(next)*: MTP head → E4B speculative decoding
  with EAGLE-2-style dynamic tree verification. Reproduce the ~2.6–3× lossless
  floor. **Gating task:** mainline llama.cpp lacks `--mtp-head`; needs an
  MTP-capable fork build (`ik_llama.cpp`/`reffdev`).
- **Phase 2 — telescoping qualifier (C1/C2)**: insert the nested-shell staircase
  as a byte-incremental fused kernel between MTP and E4B. Lossless. Contingent on
  the Phase-0c byte test (E2B `c` ≤ 0.55, fused FFN byte-savings ≥ 25%).
- **Phase 3 — entropy-gated verification-free runs (C3) + PLE prefetch (C4)**:
  conformal calibration with user-tunable δ; the lossy lever toward ~4–5×.
- **Phase 4 — online single-user self-distillation (C5)**: per-session LoRA on
  the MTP head + adaptive thresholds; lift acceptance over a session.

Kill-criteria and thresholds are inherited verbatim from the TMS doc.
