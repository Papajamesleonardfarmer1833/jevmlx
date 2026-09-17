#!/usr/bin/env python3
"""Browser-action latency lab: a toy email-triage app with the local model in the loop.

    python evals/browser_lab/app.py --port 8765 [--workers 1|2]

GET  /            -> the test page
POST /api/triage  -> one decision per queued email; the page then "performs" each
                     recommended action in the DOM and measures wall time per action
GET  /api/stats   -> server-side timings seen so far (JSON)

Workers: 1 = one model process; 2 = two model processes behind a round-robin queue,
which is what the CUDA scaling benchmark says this 8 GB card can still take.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue as queue_mod
import statistics
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for candidate in (REPO, os.path.join(REPO, "..", "parallel-decisions", "src")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

MODEL = os.environ.get("LAB_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

EMAILS = [
    {
        "id": 1,
        "from": "billing@vendor-a.com",
        "subject": "Invoice #4821 overdue",
        "body": "Your invoice 4,820 EUR is 14 days overdue. Pay today to avoid late fees.",
    },
    {
        "id": 2,
        "from": "security@bank.example",
        "subject": "Unusual sign-in blocked",
        "body": "We blocked a sign-in from a new device in another country. "
        "Not you? Reset now: http://bank-verify.example",
    },
    {
        "id": 3,
        "from": "ceo@company.com",
        "subject": "Urgent: wire before 3pm",
        "body": "I'm in a meeting, can't talk. Wire 24,500 EUR to the account I sent "
        "yesterday. Keep this between us.",
    },
    {
        "id": 4,
        "from": "newsletter@devweekly.io",
        "subject": "This week in Python",
        "body": "Top links: walrus operator tricks, GIL removal update, 5 FastAPI patterns.",
    },
    {
        "id": 5,
        "from": "hr@company.com",
        "subject": "Update your bank details",
        "body": "Payroll needs your IBAN re-confirmed by Friday. Reply with your full IBAN.",
    },
    {
        "id": 6,
        "from": "support@saas-tools.com",
        "subject": "Your ticket #9102",
        "body": "Your requested feature (CSV export) shipped in v2.14. Nothing to do.",
    },
    {
        "id": 7,
        "from": "no-reply@parcel-trk.info",
        "subject": "Parcel held: 1.99 EUR fee",
        "body": "Your package is held at the depot. Pay the 1.99 EUR customs fee via the link "
        "within 24h or it is returned.",
    },
    {
        "id": 8,
        "from": "admin@company.com",
        "subject": "Server maintenance window",
        "body": "staging-db restarts Sunday 02:00-02:30 UTC. No action needed.",
    },
    {
        "id": 9,
        "from": "dpo@company.com",
        "subject": "DSAR: export my data",
        "body": "A customer requested a copy of all personal data we hold. "
        "Legal deadline: 30 days.",
    },
    {
        "id": 10,
        "from": "sales@newvendor-b.net",
        "subject": "Quote attached - review?",
        "body": "Quote for 12,900 EUR attached. Note our bank details changed last week, "
        "please use the new IBAN.",
    },
]

SCHEMA = {
    "category": {
        "type": "enum",
        "choices": ["PHISHING", "FRAUD", "IT", "HR", "BILLING", "NEWSLETTER", "SUPPORT"],
        "description": "what this email is",
    },
    "severity": {
        "type": "enum",
        "choices": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "description": "how urgent a human response is",
    },
    "recommended_action": {
        "type": "enum",
        "choices": ["ARCHIVE", "REVIEW", "BLOCK_SENDER", "PAY", "REPLY"],
        "description": "the one browser action to perform next",
    },
}


def _worker_loop(model: str, jobs, results) -> None:
    """One model process: loads once, answers triage requests forever."""
    from parallel_decisions import Decider

    decider = Decider(model_id=model)  # backend auto-resolves (torch/CUDA here)
    schema = SCHEMA
    decider.decide("warmup context", {"flag": {"type": "boolean", "description": "warmup"}})
    while True:
        item = jobs.get()
        if item is None:
            return
        seq, context = item
        t0 = time.perf_counter()
        result = decider.decide(context, schema)
        results.put(
            (
                seq,
                {
                    "decision": result.json(),
                    "full": result.full_json(),
                    "model_ms": round(result.latency_ms, 1),
                    "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
                },
            )
        )


class ModelPool:
    """Round-robin over N model processes; each request gets its own reply queue."""

    def __init__(self, workers: int):
        self.workers = workers
        self._jobs = mp.Queue()
        self._results = mp.Queue()
        self._procs = [
            mp.Process(target=_worker_loop, args=(MODEL, self._jobs, self._results), daemon=True)
            for _ in range(workers)
        ]
        for p in self._procs:
            p.start()
        self._rr = 0
        self._rr_lock = threading.Lock()
        self._pending: dict[int, queue_mod.Queue] = {}
        self._pending_lock = threading.Lock()
        threading.Thread(target=self._collector, daemon=True).start()

    def _collector(self) -> None:
        while True:
            try:
                seq, payload = self._results.get()
            except (EOFError, OSError):  # pragma: no cover - shutdown
                return
            with self._pending_lock:
                reply = self._pending.pop(seq, None)
            if reply is not None:
                reply.put(payload)

    def decide(self, context: str) -> dict:
        with self._rr_lock:
            self._rr += 1
            seq = self._rr
        reply = queue_mod.Queue()
        with self._pending_lock:
            self._pending[seq] = reply
        self._jobs.put((seq, context))
        return reply.get(timeout=300)


class Handler(BaseHTTPRequestHandler):
    lab: ModelPool | None = None  # set in main()

    def _send(self, code: int, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif self.path == "/api/emails":
            self._send(200, EMAILS)
        elif self.path == "/api/stats":
            self._send(200, self.lab.stats() if self.lab else {})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/triage":
            self._send(404, {"error": "use POST /api/triage"})
            return
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        email = next((e for e in EMAILS if e["id"] == body.get("id")), None)
        if email is None:
            self._send(400, {"error": "unknown email id"})
            return
        context = f"From: {email['from']}\nSubject: {email['subject']}\n\n{email['body']}"
        t_server = time.perf_counter()
        payload = self.lab.decide(context)
        payload["server_ms"] = round((time.perf_counter() - t_server) * 1000, 1)
        payload["email_id"] = email["id"]
        self._send(200, payload)

    def log_message(self, fmt, *args):  # quieter
        return


class Lab:
    """Server-side sample recorder (pure bookkeeping, no model)."""

    def __init__(self, pool: ModelPool):
        self.pool = pool
        self._samples: list[dict] = []
        self._lock = threading.Lock()

    def decide(self, context: str) -> dict:
        return self.pool.decide(context)

    def record(self, payload: dict) -> None:
        with self._lock:
            self._samples.append(payload)

    def stats(self) -> dict:
        with self._lock:
            rows = list(self._samples)
        if not rows:
            return {"workers": self.pool.workers, "requests": 0}
        walls = [r["wall_ms"] for r in rows]
        return {
            "workers": self.pool.workers,
            "requests": len(rows),
            "p50_wall_ms": round(statistics.median(walls), 1),
            "p95_wall_ms": round(sorted(walls)[int(0.95 * len(walls)) - 1], 1),
            "mean_model_ms": round(statistics.mean(r["model_ms"] for r in rows), 1),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    pool = ModelPool(args.workers)
    lab = Lab(pool)
    Handler.lab = lab
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"lab on http://127.0.0.1:{args.port}  workers={args.workers} model={MODEL}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
