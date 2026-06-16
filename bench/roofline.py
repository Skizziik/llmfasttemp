"""Phase-0a: byte-accurate roofline for the Gemma 4 E4B Q4_K_M GGUF on RTX 4060.

Reads the real per-tensor byte sizes straight from the GGUF (no model load) and
computes how many bytes must stream from VRAM per generated token, hence the
theoretical memory-bound throughput ceiling. Comparing that ceiling to the
measured baseline (62.12 tok/s) tells us how close llama.cpp already is to the
memory wall -- i.e. how little headroom there is for anything *except*
emitting more tokens per weight-read (speculative decoding). This is the
quantitative justification for the whole TMS plan.

Usage:  python bench/roofline.py
"""

from __future__ import annotations

import os
import sys
from gguf import GGUFReader

# Windows consoles default to cp1251 here; force UTF-8 so unicode prints work.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

GGUF = os.environ.get(
    "PRESTO_GGUF", r"D:/DEV/Claude/UNITYGAMES/weights/llm/gemma-4-E4B-it-Q4_K_M.gguf"
)

# RTX 4060 (AD107): 128-bit GDDR6 @ 17 Gbps = 272 GB/s raw. NVIDIA claims the
# 24 MB L2 makes *effective* bandwidth ~equivalent to ~453 GB/s on Ampere.
BW_RAW = 272e9
BW_EFF = 453e9
MEASURED_TOK_S = 62.12  # from results/ baseline run


def classify(name: str) -> str:
    n = name.lower()
    if "token_embd" in n or ("embd" in n and "blk" not in n):
        return "input_embedding"
    if "output" in n and "norm" not in n:
        return "output_proj"
    if "per_layer" in n or "ple" in n or "altup" in n:
        return "ple/altup"
    if n.startswith("blk.") or ".blk." in n or "blk." in n:
        return "transformer_blocks"
    return "other"


def main() -> None:
    r = GGUFReader(GGUF)
    groups: dict[str, int] = {}
    total = 0
    tied_embed_bytes = 0
    for t in r.tensors:
        b = int(t.n_bytes)
        total += b
        g = classify(t.name)
        groups[g] = groups.get(g, 0) + b
        if g == "input_embedding":
            tied_embed_bytes = b

    has_separate_output = groups.get("output_proj", 0) > 0
    # Gemma ties input embeddings to the output projection. If there is no
    # separate output tensor, the embedding matrix IS read once per token to
    # produce logits over the full vocab.
    per_token = groups.get("transformer_blocks", 0) + groups.get("ple/altup", 0)
    if has_separate_output:
        per_token += groups["output_proj"]
        note_out = "separate output_proj tensor read per token"
    else:
        per_token += tied_embed_bytes
        note_out = "tied embeddings: token_embd read per token for logits"

    GB = 1024 ** 3
    print(f"GGUF: {os.path.basename(GGUF)}")
    print(f"total weight bytes: {total/GB:.3f} GiB\n")
    print("by group:")
    for g, b in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {g:20s} {b/GB:7.3f} GiB  ({100*b/total:5.1f}%)")
    print(f"\noutput handling: {note_out}")
    print(f"\nweight bytes READ PER DECODE TOKEN: {per_token/GB:.3f} GiB")

    ceil_raw = BW_RAW / per_token
    ceil_eff = BW_EFF / per_token
    print("\n--- memory-bound ceilings (weights only, KV ignored) ---")
    print(f"  @ {BW_RAW/1e9:.0f} GB/s (raw):       {ceil_raw:6.1f} tok/s")
    print(f"  @ {BW_EFF/1e9:.0f} GB/s (L2-eff):    {ceil_eff:6.1f} tok/s")
    print(f"  measured baseline:          {MEASURED_TOK_S:6.1f} tok/s")
    eff_raw = 100 * MEASURED_TOK_S / ceil_raw
    eff_eff = 100 * MEASURED_TOK_S / ceil_eff
    print(f"\n  measured / raw ceiling:    {eff_raw:5.1f}%  (hardware efficiency η vs raw BW)")
    print(f"  measured / L2-eff ceiling: {eff_eff:5.1f}%")
    print(f"\n  implied effective BW used: {MEASURED_TOK_S*per_token/1e9:.0f} GB/s")
    print("\nReading: the baseline already runs near the memory wall, so the only")
    print("large lever is verified-tokens-per-weight-read (speculative decoding).")


if __name__ == "__main__":
    main()
