"""Presto evaluation harness.

Runs the fixed prompt suite (bench/prompts.json) against a backend, capturing
for every prompt: the full answer, tokens generated, tok/s, latency, and the
GPU VRAM the model occupies. Writes a detailed Markdown report under results/
and appends a one-line summary row to BENCHMARKS.md (the progress journal).

Currently supports the llama.cpp reference backend by launching `llama-server`
from the official prebuilt CUDA build. The real Presto engine will plug in here
as another runner with the same `run_prompt` contract.

Usage:
    python bench/run_suite.py --label "baseline-llamacpp-b9672"

Env / flags:
    PRESTO_GGUF        path to the GGUF (default: the Unity weights path)
    PRESTO_LLAMA_BIN   dir containing llama-server.exe (default: d:/tmp/llamacpp)
    --port, --ngl, --ctx
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
PROMPTS_PATH = os.path.join(ROOT, "bench", "prompts.json")
RESULTS_DIR = os.path.join(ROOT, "results")
JOURNAL = os.path.join(ROOT, "BENCHMARKS.md")

DEFAULT_GGUF = os.environ.get(
    "PRESTO_GGUF", r"D:/DEV/Claude/UNITYGAMES/weights/llm/gemma-4-E4B-it-Q4_K_M.gguf"
)
LLAMA_BIN_DIR = os.environ.get("PRESTO_LLAMA_BIN", r"D:/tmp/llamacpp")


# --- GPU telemetry -------------------------------------------------------
def gpu_mem_used_mib() -> int | None:
    """Total VRAM currently used on GPU 0, in MiB (via nvidia-smi)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
             "--id=0"],
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


# --- llama.cpp server runner --------------------------------------------
class LlamaServerRunner:
    def __init__(self, gguf: str, port: int, ngl: int, ctx: int,
                 bin_dir: str = LLAMA_BIN_DIR, extra_args: list | None = None):
        self.gguf, self.port, self.ngl, self.ctx = gguf, port, ngl, ctx
        self.bin_dir = bin_dir
        self.extra_args = extra_args or []
        self.proc = None
        self.base = f"http://127.0.0.1:{port}"
        self.build = "unknown"
        self.mem_idle = None
        self.mem_loaded = None

    def __enter__(self):
        exe = os.path.join(self.bin_dir, "llama-server.exe")
        self.mem_idle = gpu_mem_used_mib()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self._logpath = os.path.join(RESULTS_DIR, "_llama_server.log")
        # Redirect server output to a file (avoids PIPE buffer deadlock since we
        # never drain it) and lets us read the build string back.
        self._logf = open(self._logpath, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            # --jinja enables the full chat template so chat_template_kwargs
            # (enable_thinking=false) is honored. extra_args carries MTP flags
            # (--mtp-head <drafter> --spec-type mtp) for the speculative runs.
            [exe, "-m", self.gguf, "--port", str(self.port),
             "-ngl", str(self.ngl), "-c", str(self.ctx), "--no-webui", "--jinja"]
            + self.extra_args,
            stdout=self._logf, stderr=subprocess.STDOUT, cwd=self.bin_dir,
        )
        self._wait_ready()
        self.mem_loaded = gpu_mem_used_mib()
        self._read_build()
        return self

    def _read_build(self):
        try:
            with open(self._logpath, encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("build:") or "build: " in line:
                        self.build = line.split("build:")[-1].strip().split(" ")[0]
                        return
        except Exception:
            pass

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if getattr(self, "_logf", None):
            self._logf.close()

    def _wait_ready(self, timeout: float = 180.0):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited early; check the build/CUDA")
            try:
                with urllib.request.urlopen(self.base + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return
            except Exception:
                time.sleep(1.0)
        raise TimeoutError("llama-server did not become ready in time")

    def run_prompt(self, prompt: str, max_tokens: int) -> dict:
        # Chat endpoint so llama-server applies Gemma's own chat template from
        # the GGUF -> correct behavior and fair answer-quality comparison.
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0,
            # Disable Gemma 4's chain-of-thought: answer directly, no separate
            # reasoning channel, deterministic and comparable across versions.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        peak = gpu_mem_used_mib()
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read())
        peak = max(peak or 0, gpu_mem_used_mib() or 0)
        tim = data.get("timings", {})
        choice = data["choices"][0]
        msg = choice["message"]
        answer = (msg.get("content") or "").strip()
        # Gemma 4 is a reasoning model: chain-of-thought lands in reasoning_content,
        # the final answer in content. Capture both for honest quality comparison.
        reasoning = (msg.get("reasoning_content") or "").strip()
        tokens = int(tim.get("predicted_n") or data.get("usage", {}).get("completion_tokens", 0))
        ms = round(float(tim.get("predicted_ms", 0.0)), 1)
        tok_s = round(tokens / (ms / 1000), 2) if ms > 0 else 0.0
        # Speculative-decoding telemetry (present only on MTP/draft runs). Field
        # names vary across builds, so probe the likely ones.
        draft_n = tim.get("draft_n") or tim.get("n_draft")
        draft_acc = tim.get("draft_n_accepted") or tim.get("n_draft_accepted")
        accept_rate = (round(draft_acc / draft_n, 3)
                       if draft_n and draft_acc is not None else None)
        return {"answer": answer, "reasoning": reasoning, "tokens": tokens,
                "tok_s": tok_s, "ms": ms, "peak_vram_mib": peak,
                "finish": choice.get("finish_reason", ""),
                "draft_n": draft_n, "draft_accepted": draft_acc,
                "accept_rate": accept_rate}


# --- report writing ------------------------------------------------------
def write_report(label: str, meta: dict, rows: list[dict]) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = meta["utc"].replace(":", "").replace("-", "").replace(" ", "_")[:15]
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label)
    path = os.path.join(RESULTS_DIR, f"{stamp}_{safe}.md")

    # Aggregate speed = total tokens / total time. This is the true throughput
    # and is immune to short-prompt outliers (1-token answers etc.).
    total_tokens = sum(r["tokens"] for r in rows)
    total_s = sum(r["ms"] for r in rows) / 1000.0
    avg_tok_s = round(total_tokens / total_s, 2) if total_s > 0 else 0.0

    L = []
    L.append(f"# Eval run — {label}\n")
    L.append(f"- **UTC:** {meta['utc']}")
    L.append(f"- **Backend:** {meta['backend']} (build {meta['build']})")
    L.append(f"- **Hardware:** {meta['gpu']}")
    L.append(f"- **Model:** {os.path.basename(meta['gguf'])}")
    L.append(f"- **Settings:** ngl={meta['ngl']}, ctx={meta['ctx']}, temperature=0.0 (greedy)")
    if meta.get("mem_loaded") is not None:
        delta = (meta["mem_loaded"] - (meta["mem_idle"] or 0))
        L.append(f"- **VRAM (model resident):** ~{delta} MiB "
                 f"(idle {meta['mem_idle']} → loaded {meta['mem_loaded']} MiB)")
    L.append(f"- **Avg generation speed:** **{avg_tok_s} tok/s** across {len(rows)} prompts\n")

    truncated = [r["id"] for r in rows if r.get("finish") == "length"]
    if truncated:
        L.append(f"> ⚠️ Truncated at token cap (raise max_tokens): {', '.join(truncated)}\n")

    has_accept = any(r.get("accept_rate") is not None for r in rows)
    L.append("## Summary\n")
    acc_h = " accept |" if has_accept else ""
    acc_sep = "-------:|" if has_accept else ""
    L.append(f"| # | id | category | entropy | tokens | tok/s |{acc_h} latency (ms) | finish | peak VRAM (MiB) |")
    L.append(f"|---|----|----------|---------|-------:|------:|{acc_sep}-------------:|--------|----------------:|")
    for i, r in enumerate(rows, 1):
        acc_c = f" {r.get('accept_rate')} |" if has_accept else ""
        L.append(f"| {i} | {r['id']} | {r['category']} | {r['entropy']} | "
                 f"{r['tokens']} | {r['tok_s']} |{acc_c} {r['ms']} | {r.get('finish','')} | {r['peak_vram_mib']} |")
    L.append("")

    L.append("## Full answers (for quality comparison across versions)\n")
    for i, r in enumerate(rows, 1):
        L.append(f"### {i}. `{r['id']}` — {r['category']} ({r['tok_s']} tok/s, {r['tokens']} tok)\n")
        L.append(f"**Prompt:** {r['prompt']}\n")
        if r.get("reasoning"):
            L.append("<details><summary>reasoning (CoT)</summary>\n")
            L.append("```")
            L.append(r["reasoning"])
            L.append("```")
            L.append("</details>\n")
        L.append("**Answer:**\n")
        L.append("```")
        L.append(r["answer"])
        L.append("```\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path, avg_tok_s, total_tokens


def append_journal(label: str, meta: dict, avg_tok_s: float, report_path: str):
    header = (
        "# Presto — benchmark journal\n\n"
        "One row per engine milestone. The point of this file: track exactly how "
        "much each technique added (tok/s and % vs the previous best), with a link "
        "to the full per-prompt report. This is the data a write-up/paper draws on.\n\n"
        "| Date (UTC) | Label | Technique added | Avg tok/s | Δ vs baseline | Report |\n"
        "|------------|-------|-----------------|----------:|--------------:|--------|\n"
    )
    if not os.path.exists(JOURNAL):
        with open(JOURNAL, "w", encoding="utf-8") as f:
            f.write(header)

    # Baseline delta: read first data row's tok/s if present.
    baseline = None
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            if line.startswith("| 20"):  # a data row starting with a date
                try:
                    baseline = float(line.split("|")[4])
                    break
                except (IndexError, ValueError):
                    pass
    if baseline:
        delta = f"+{round((avg_tok_s / baseline - 1) * 100, 1)}%"
    else:
        delta = "— (baseline)"

    rel = os.path.relpath(report_path, ROOT).replace("\\", "/")
    row = (f"| {meta['utc']} | {label} | {meta.get('technique','—')} | "
           f"{avg_tok_s} | {delta} | [report]({rel}) |\n")
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="short name for this run/milestone")
    ap.add_argument("--technique", default="baseline (llama.cpp, autoregressive)")
    ap.add_argument("--build", default="b9672", help="engine build id for the report")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--llama-bin", default=LLAMA_BIN_DIR,
                    help="dir with llama-server.exe (use the MTP fork build for MTP runs)")
    ap.add_argument("--mtp-head", default=None,
                    help="path to the gemma4_assistant drafter GGUF; enables MTP spec decoding")
    ap.add_argument("--spec-type", default="mtp", help="speculation type when --mtp-head is set")
    args = ap.parse_args()

    extra_args = []
    if args.mtp_head:
        extra_args += ["--mtp-head", args.mtp_head, "--spec-type", args.spec_type]

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        suite = json.load(f)

    gpu_name = "unknown GPU"
    try:
        gpu_name = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader", "--id=0"],
            text=True).strip()
    except Exception:
        pass

    print(f"Starting llama-server on port {args.port} "
          f"{'(MTP spec decoding)' if args.mtp_head else '(autoregressive)'} ...")
    rows = []
    with LlamaServerRunner(args.gguf, args.port, args.ngl, args.ctx,
                           bin_dir=args.llama_bin, extra_args=extra_args) as runner:
        print(f"Model loaded (VRAM {runner.mem_loaded} MiB). Running {len(suite['prompts'])} prompts...")
        for p in suite["prompts"]:
            print(f"  -> {p['id']} ...", end="", flush=True)
            res = runner.run_prompt(p["prompt"], p["max_tokens"])
            res.update({"id": p["id"], "category": p["category"],
                        "entropy": p["entropy"], "prompt": p["prompt"]})
            rows.append(res)
            acc = f", accept {res['accept_rate']}" if res.get("accept_rate") is not None else ""
            print(f" {res['tok_s']} tok/s, {res['tokens']} tok{acc}")
        meta = {
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "backend": "llama.cpp server",
            "build": runner.build if runner.build != "unknown" else args.build,
            "gpu": gpu_name,
            "gguf": args.gguf, "ngl": args.ngl, "ctx": args.ctx,
            "mem_idle": runner.mem_idle, "mem_loaded": runner.mem_loaded,
            "technique": args.technique,
        }

    path, avg, total = write_report(args.label, meta, rows)
    append_journal(args.label, meta, avg, path)
    print(f"\nReport: {os.path.relpath(path, ROOT)}")
    print(f"Avg: {avg} tok/s | total {total} tokens generated")
    print(f"Journal updated: BENCHMARKS.md")


if __name__ == "__main__":
    main()
