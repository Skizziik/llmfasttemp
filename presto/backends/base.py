"""The one interface every Presto backend implements.

The whole point of this abstraction: the HTTP server and the browser UI are
written once, against `Backend`, and never change as we swap mock -> reference
-> the real speculative-decoding engine. That also lets the playground put two
backends side by side and benchmark them in the same chat window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Token:
    """A single streamed token plus optional telemetry.

    `accepted` and `drafted` exist so that, once speculative decoding lands, the
    playground can visualize how many draft tokens the verifier accepted on the
    step that produced this token. Mock/reference backends just leave them at 0.
    """

    text: str
    drafted: int = 0   # how many tokens the drafter proposed this step
    accepted: int = 0  # how many of those the verifier accepted
    meta: dict = field(default_factory=dict)


class Backend:
    """Base class for inference engines.

    Subclasses override `generate` to yield `Token`s one at a time. Yielding (as
    opposed to returning a string) is what makes browser streaming + live
    tokens/sec possible.
    """

    name: str = "base"

    def generate(self, prompt: str, max_tokens: int = 256, **opts) -> Iterator[Token]:
        raise NotImplementedError

    def info(self) -> dict:
        """Metadata the UI shows (backend name, model, whether it's real)."""
        return {"name": self.name, "real": False}
