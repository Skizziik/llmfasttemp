# Presto 🎵

A from-scratch LLM inference engine built for **one thing**: making small Gemma-4
models feel *instant* on a single consumer GPU.

Presto is not a fork of llama.cpp. It's a new runtime whose **architecture** —
speculative decoding baked into the core from day one — is the speed lever, with
CUDA used purely as the compute technology underneath.

Target model: [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it)
(MatFormer, effective ~4B params, fits in 8 GB VRAM with no CPU offload).

> **Status: Phase 0 — playground scaffold.** The browser chat UI and local
> server run *today* on a zero-dependency mock backend so the full UX loop is
> real. The actual Presto engine is built into this harness phase by phase
> (see [ARCHITECTURE.md](ARCHITECTURE.md)), and benchmarked live in the same UI.

## The core idea (why this can be fast)

Single-stream decoding is **memory-bandwidth bound**: every token re-reads the
whole model from memory. The only way to beat that is to emit *several* tokens
per weight-read — **speculative decoding**. Our headline bet:

- **Matryoshka self-speculation** — E4B *contains* E2B as a strict weight subset
  (MatFormer/FFN nesting). So E2B drafts and E4B verifies **from the same weights
  in memory** — zero extra VRAM for the drafter, perfect vocab alignment, high
  acceptance. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Run the playground (now, zero deps)

Requires only Python 3.9+ (stdlib only — no `pip install`):

```bash
python -m presto.server
# then open http://localhost:8000 in your browser
```

You'll get a chat UI with a live tokens/sec readout. The backend is a mock until
the engine lands — but the loop, streaming, and benchmark plumbing are real.

## Layout

```
presto/
  server.py          # stdlib HTTP server + SSE token streaming
  backends/
    base.py          # Backend interface — what every engine must implement
    mock.py          # Phase-0 mock backend (no deps, streams canned text)
web/
  index.html         # browser chat playground with live tok/s
ARCHITECTURE.md      # the design north-star: phases, math, innovations
```
