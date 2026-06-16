"""Presto playground server — stdlib only, no pip install required.

Serves the browser chat UI and streams generated tokens over Server-Sent Events
(SSE). The backend is swappable: today it's the mock; later it's the real
engine. Run with:

    python -m presto.server
    # open http://localhost:8000

The token-streaming protocol (one JSON object per SSE `data:` line):
    {"t": "<text>", "drafted": 0, "accepted": 0}   # a token
    {"done": true, "tokens": N, "elapsed": S, "tok_s": X}  # final stats
"""

from __future__ import annotations

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .backends.mock import MockBackend

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.normpath(os.path.join(HERE, "..", "web"))


def _make_backend():
    """Select backend via PRESTO_BACKEND (mock | llamacpp). Defaults to mock so
    the playground always runs with zero setup; set llamacpp to chat with the
    real model through a running llama-server (see backends/llamacpp.py)."""
    kind = os.environ.get("PRESTO_BACKEND", "mock").lower()
    if kind == "llamacpp":
        from .backends.llamacpp import LlamaCppBackend
        return LlamaCppBackend()
    return MockBackend()


BACKEND = _make_backend()


class Handler(BaseHTTPRequestHandler):
    # Quieter logs (one line per request is enough).
    def log_message(self, fmt, *args):
        print("[presto] " + (fmt % args))

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send_file(os.path.join(WEB_DIR, "index.html"), "text/html")
        if path == "/api/info":
            return self._send_json(BACKEND.info())
        return self._send_error(404, "not found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/generate":
            return self._send_error(404, "not found")

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return self._send_error(400, "bad json")

        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 256))

        # SSE headers.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        n, start = 0, time.perf_counter()
        try:
            for tok in BACKEND.generate(prompt, max_tokens=max_tokens):
                n += 1
                payload = {"t": tok.text, "drafted": tok.drafted, "accepted": tok.accepted}
                self._sse(payload)
            elapsed = time.perf_counter() - start
            self._sse({
                "done": True,
                "tokens": n,
                "elapsed": round(elapsed, 3),
                "tok_s": round(n / elapsed, 2) if elapsed > 0 else 0.0,
            })
        except (BrokenPipeError, ConnectionResetError):
            # Client navigated away / hit stop — fine, just end the stream.
            pass

    # --- helpers ---------------------------------------------------------
    def _sse(self, obj: dict):
        self.wfile.write(b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def _send_json(self, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_file(self, path: str, ctype: str):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return self._send_error(404, f"missing {os.path.basename(path)}")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, msg: str):
        data = msg.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(host: str = "127.0.0.1", port: int = 8000):
    server = ThreadingHTTPServer((host, port), Handler)
    info = BACKEND.info()
    print(f"Presto playground — backend: {info['name']} (real={info['real']})")
    print(f"  open http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()


if __name__ == "__main__":
    port = int(os.environ.get("PRESTO_PORT", "8000"))
    main(port=port)
