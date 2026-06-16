"""Pluggable inference backends.

Every engine — mock, reference, and the real Presto core — implements the same
`Backend` interface from `base.py`, so the server and UI never change as we swap
in faster engines.
"""

from .base import Backend, Token

__all__ = ["Backend", "Token"]
