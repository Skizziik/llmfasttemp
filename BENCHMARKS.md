# Presto — benchmark journal

One row per engine milestone. The point of this file: track exactly how much each technique added (tok/s and % vs the previous best), with a link to the full per-prompt report. This is the data a write-up/paper draws on.

| Date (UTC) | Label | Technique added | Avg tok/s | Δ vs baseline | Report |
|------------|-------|-----------------|----------:|--------------:|--------|
| 2026-06-16 23:30:51 | baseline-llamacpp-b9672 | baseline (llama.cpp b9672, autoregressive, Q4_K_M, thinking off) | 62.12 | — (baseline) | [report](results/20260616_233051_baseline-llamacpp-b9672.md) |
| 2026-06-16 23:57:26 | fa-off-baseline | plain decode, flash-attn OFF (MTP-fork build, control) | 57.52 | -7.4% | [report](results/20260616_235726_fa-off-baseline.md) |
| 2026-06-16 23:57:55 | mtp-spec-gemma4assistant | MTP spec decoding (gemma4_assistant Q4_K_M drafter, FA off) | 80.28 | +29.2% | [report](results/20260616_235755_mtp-spec-gemma4assistant.md) |
