"""`openjev serve`: HTTP wrapper around the parallel decision engine.

    openjev serve --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --port 8000

POST /decide  {"schema": {...}, "context": "...", "temperature": 1.0}
GET  /health  -> {"ok": true, "model": M}
"""

from __future__ import annotations

import json
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable, Optional


def make_handler(decide_fn: Callable[[dict, str, Optional[float]], dict], model_id: str):
    """Build a request handler around decide_fn(schema_dict, context, temperature).

    decide_fn is the seam tests use: a callable that returns the result dict
    (or raises ValueError for schema errors, anything else for 500s).
    """

    class Handler(BaseHTTPRequestHandler):
        # ponytail: serial server, one Metal GPU; queue/batching if concurrency matters

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "model": model_id})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/decide":
                self._send(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
                schema_dict = payload["schema"]
                context = payload["context"]
                temperature = payload.get("temperature", 1.0)
            except (json.JSONDecodeError, KeyError) as e:
                self._send(400, {"error": f"bad request: {e}"})
                return
            try:
                result = decide_fn(schema_dict, context, temperature)
            except ValueError as e:  # StructuredSchema schema errors
                self._send(400, {"error": str(e)})
                return
            except Exception as e:
                traceback.print_exc()  # full traceback to stderr
                first_line = str(e).splitlines() or [type(e).__name__]
                self._send(500, {"error": first_line[0]})
                return
            self._send(200, result)

    return Handler


def serve(model_id: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Load the model once, then serve decisions until interrupted."""
    from openjev.engine import load_engine, run_parallel_generation
    from openjev.schema import StructuredSchema

    print(f"Loading {model_id} ...", flush=True)
    model, tokenizer = load_engine(model_id)

    def decide_fn(schema_dict: dict, context: str, temperature: Optional[float] = 1.0) -> dict:
        schema = StructuredSchema(schema_dict)
        return run_parallel_generation(model, tokenizer, context, schema,
                                       temperature=temperature)

    server = HTTPServer((host, port), make_handler(decide_fn, model_id))
    print(f"openjev serving {model_id} on http://{host}:{port}", flush=True)
    server.serve_forever()
