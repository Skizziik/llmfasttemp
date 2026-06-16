"""Reference backend: stream real Gemma 4 tokens from a running llama-server.

This makes the browser playground talk to the actual model (the baseline we must
beat), not the mock. It connects to an already-running llama-server's
OpenAI-compatible streaming endpoint and relays tokens.

Start a server first (mainline works for plain decode; the MTP fork adds
--mtp-head for speculative decoding), e.g.:

    llama-server -m gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -c 4096 --jinja --port 8080

then run the playground with:

    PRESTO_BACKEND=llamacpp PRESTO_LLAMA_URL=http://127.0.0.1:8080 python -m presto.server
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Iterator

from .base import Backend, Token


class LlamaCppBackend(Backend):
    name = "llamacpp"

    def __init__(self, url: str | None = None):
        self.url = (url or os.environ.get("PRESTO_LLAMA_URL", "http://127.0.0.1:8080")).rstrip("/")

    def generate(self, prompt: str, max_tokens: int = 256, **opts) -> Iterator[Token]:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": float(opts.get("temperature", 0.0)),
            "stream": True,
            # Keep parity with the benchmark: Gemma 4 thinking disabled.
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        req = urllib.request.Request(
            self.url + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield Token(text=text)

    def info(self) -> dict:
        return {"name": self.name, "real": True, "model": f"llama-server @ {self.url}"}
