# Presto — Research Note: where the speed actually hides

Senior-level survey of the LLM-inference acceleration literature (as of mid-2026),
written for one concrete target: **Gemma 4 E4B (MatFormer) on a single 8 GB GPU,
batch size 1**. The goal is to find a genuinely unoccupied idea that fits *this*
regime — and to be honest about what is already known vs. what is our bet.

---

## 1. The economics we are optimizing

Batch-size-1 decode is **memory-bandwidth bound**: to emit one token the GPU must
stream the model's weights (and KV-cache) from VRAM once. Compute units sit
mostly idle. So:

```
tok/s ≈ (VRAM bandwidth / bytes read per token) × tokens emitted per weight-read
```

- You cannot grow the first factor much — llama.cpp already saturates bandwidth.
- **Every acceleration method that matters increases the second factor**: emit
  more accepted tokens per pass over the weights. That is speculative decoding,
  full stop.

A second, subtler cost matters at bs=1 and is usually ignored: **redundant weight
reads**. If two phases read overlapping weight tensors from VRAM separately, you
pay the bandwidth twice. Hold this thought — it is where our gap lives.

---

## 2. Field map — every method, and which axis it exploits

| Method | Core trick | Axis exploited | Status |
|---|---|---|---|
| Vanilla speculative decoding | separate small draft model verifies in bulk | external model | known |
| **EAGLE-1/2/3** | draft on the target's *feature/hidden states*, not tokens; dynamic draft **trees** | feature + tree | SOTA, known |
| **Medusa** | extra parallel prediction heads on the base model | feature | known |
| **LayerSkip / Draft&Verify** | early-exit at layer *K* drafts; full depth verifies; **reuses draft activations along depth** | **depth** | known (ACL'24) |
| **MTP (DeepSeek-V3, Gemma 4)** | trained heads predict N future tokens, share embedding + last-layer activations | feature | known, shipped |
| **MatFormer self-spec** | E2B (a width-subset of E4B) drafts, E4B verifies | **width** | *mentioned* in MatFormer paper; not productized |
| **Speculative cascades** (Google, 2025) | replace hard accept/reject with a confidence **deferral rule** | confidence | known |
| **Cascade Speculative Drafting** | multi-stage draft models, biggest verifies last | model cascade | known |
| **PEARL** | run draft and target **concurrently**, pipelined | scheduling | known |
| **Lookahead / Jacobi** | parallel n-gram fixed-point decoding, no draft model | algorithmic | known |
| **Prompt-lookup / n-gram** | copy repeated spans for free | retrieval | known |
| Quantization / KV-quant / FlashAttn | read fewer bytes per weight | bytes-per-token | known |

**Conclusion:** depth (LayerSkip), feature (EAGLE/MTP), confidence (cascades),
scheduling (PEARL), and the external-model axis are all occupied. Even the bare
"E2B drafts E4B" width idea is foreseen in the MatFormer paper.

So the bare Matryoshka-self-speculation idea from our first design doc is **good
engineering but not novel**. We need to go one level deeper.

---

## 3. The unoccupied cell

Two facts nobody has combined:

1. **MatFormer gives a *free, zero-memory ladder*.** Mix'n'Match extracts
   *hundreds* of nested submodels (≈582M…full) from one weight set, with no extra
   parameters and no model loading — switching "draft size" is just masking more
   or fewer FFN neurons in the *same resident weights*.
2. **At bs=1 the dominant hidden cost is redundant weight reads**, and in nested
   self-speculation the small model's weights are a *physical subset* of the big
   model's. The draft reads them; the verify reads them again.

Everyone else picks a *fixed* draft model (separate weights, or a fixed early-exit
depth). Nobody has a draft whose **size is a free continuous dial over shared
weights**, because no other architecture gives that for free. MatFormer does.

---

## 4. Our bet: **Ladder Speculation** (working name)

Use MatFormer's nested ladder as the substrate for a self-speculative scheme with
two ideas that are only viable *because* the ladder is free:

### 4.1 Granularity-adaptive drafting (zero-cost difficulty dial)
Per output segment, a confidence gate (speculative-cascade-style deferral) picks
the **draft rung**: a tiny submodel on low-entropy spans (boilerplate, code,
lists), a bigger rung where the model is uncertain. Unique property: changing the
rung costs **zero extra VRAM and zero model-load** — it's the same weights with a
different FFN mask. On 8 GB, that "free" is the whole point: every prior
adaptive-draft scheme pays memory or load latency to change draft capacity; we
pay neither.

### 4.2 Cascade verification up the ladder (fewer full-E4B reads)
Don't verify every draft with full E4B. Verify first with a mid-rung submodel;
escalate to full E4B **only on disagreement** (deferral rule). Because mid-rung
weights are a strict subset of E4B's, an escalation re-reads only the *extra*
neurons, not the whole model — turning verification cost into a graded, mostly-
amortized read instead of an all-or-nothing full pass.

### 4.3 (Stretch) Weight-tile fusion
Where the draft→verify dependency allows pipelining (PEARL-style), schedule the
shared weight tiles so a tile streamed from VRAM for verification of chunk *N* is
*also* used to advance the draft of chunk *N+1* before it leaves cache —
approaching "one weight read serves both roles." This is the deepest and least
certain part; see risks.

---

## 5. Honest risks (why this might not beat baseline)

- **Activation divergence.** E2B and E4B share the residual-stream width but
  diverge after the first FFN (E4B adds neuron contributions). So we cannot blind-
  reuse draft layer outputs as verify inputs; §4.3 fusion is limited to the
  weight-read level, not the activation level. This is the main reason the stretch
  goal is a *stretch*.
- **Gate overhead.** A per-segment confidence gate that mispredicts difficulty
  wastes passes. The gate must be cheaper than the work it saves.
- **Acceptance rate is empirical.** Whether a tiny rung drafts well enough for
  high acceptance on *our* prompts is unknown until measured. The whole bet stands
  or falls on measured acceptance length per rung.
- **bs=1 caveat (from Google's own MTP docs):** at batch size 1 the verify-overlap
  win shrinks on hardware without good parallelism. Our dense small model helps,
  but this caps the ceiling.

**What is honestly novel here:** not cascades, not self-spec, not adaptive draft
in isolation — those exist. It is using MatFormer's *free elastic ladder* as the
substrate so that **adapting draft/verify capacity is zero-cost in memory**, which
is precisely what makes graded, per-segment speculation affordable on 8 GB. That
specific combination, to the best of this survey, is not in the literature or in
any shipped engine.

---

## 6. How we prove or kill it (no hand-waving)

1. Phase 1 baseline: E4B autoregressive tok/s (the number to beat).
2. Measure **acceptance length per ladder rung** on real prompts (code, prose,
   JSON). If a small rung's acceptance is poor everywhere, §4.1 dies — report it.
3. A/B in the playground: fixed-rung self-spec vs. granularity-adaptive. The UI
   already streams `drafted`/`accepted` per token for exactly this.
4. Only if §4.1/§4.2 win do we attempt §4.3.

---

## Sources

- [LayerSkip: Early Exit & Self-Speculative Decoding (arXiv 2404.16710)](https://arxiv.org/abs/2404.16710)
- [MatFormer: Nested Transformer for Elastic Inference (arXiv 2310.07707)](https://arxiv.org/abs/2310.07707)
- [DeepSeek-V3 Technical Report (MTP)](https://arxiv.org/html/2412.19437v1)
- [Speculative cascades — Google Research](https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/)
- [Faster Cascades via Speculative Decoding (OpenReview)](https://openreview.net/pdf?id=vo9t20wsmd)
- [Cascade Speculative Drafting (arXiv 2312.11462)](https://arxiv.org/pdf/2312.11462)
- [FastMTP: Accelerating LLM Inference with Enhanced MTP (arXiv 2509.18362)](https://arxiv.org/pdf/2509.18362)
- [Gemma 4 Multi-Token Prediction overview](https://ai.google.dev/gemma/docs/mtp/overview)
- [EAGLE-3 in vLLM — Red Hat Developers](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)
