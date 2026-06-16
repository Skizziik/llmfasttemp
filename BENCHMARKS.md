# Presto — benchmark journal

One row per engine milestone. The point of this file: track exactly how much each technique added (tok/s and % vs the previous best), with a link to the full per-prompt report. This is the data a write-up/paper draws on.

| Date (UTC) | Label | Technique added | Avg tok/s | Δ vs baseline | Report |
|------------|-------|-----------------|----------:|--------------:|--------|
| 2026-06-16 23:30:51 | baseline-llamacpp-b9672 | baseline (llama.cpp b9672, autoregressive, Q4_K_M, thinking off) | 62.12 | — (baseline) | [report](results/20260616_233051_baseline-llamacpp-b9672.md) |
