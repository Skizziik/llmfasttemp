"""Phase-0 mock backend — zero dependencies, runs anywhere Python runs.

It does NOT run a model. It streams a canned, prompt-aware response token by
token at a configurable simulated speed, so the entire UX loop (typing, SSE
streaming, live tokens/sec, stop) is real and testable before the engine exists.

When the real engine lands it implements the exact same `Backend.generate`
signature and drops in with no server/UI changes.
"""

from __future__ import annotations

import time
from typing import Iterator

from .base import Backend, Token

_CANNED = (
    "Hey — this is Presto's Phase-0 mock backend talking. I'm not a real model "
    "yet: I just stream canned text so the playground's UX loop, SSE streaming, "
    "and the live tokens/sec meter are real and testable. Once the engine lands, "
    "google/gemma-4-E4B-it plugs into this exact interface and these words get "
    "replaced by genuine generation — with E2B drafting and E4B verifying from "
    "the same weights in memory. Type something and watch the counter move."
)


class MockBackend(Backend):
    name = "mock"

    def __init__(self, sim_tokens_per_sec: float = 60.0):
        # Simulated decode speed so the tok/s meter shows something believable.
        self.delay = 1.0 / sim_tokens_per_sec if sim_tokens_per_sec > 0 else 0.0

    def generate(self, prompt: str, max_tokens: int = 256, **opts) -> Iterator[Token]:
        prompt = (prompt or "").strip()
        if prompt:
            yield Token(text=f'You said: "{prompt[:120]}". ')
            if self.delay:
                time.sleep(self.delay)

        words = _CANNED.split(" ")
        for i, w in enumerate(words):
            if i >= max_tokens:
                break
            # Word-ish "tokens" with trailing spaces — good enough to stream.
            yield Token(text=w + (" " if i < len(words) - 1 else ""))
            if self.delay:
                time.sleep(self.delay)

    def info(self) -> dict:
        return {"name": self.name, "real": False, "model": "(none — canned text)"}
