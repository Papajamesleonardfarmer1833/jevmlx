"""Benchmark: torch backend per-request latency, CPU vs CUDA vs parallel processes.

E1 from the roadmap (docs/07, docs/09): run the torch port on the 8 GB-VRAM PC and
record what CUDA does to the decision latency story.

    .venv/python evals/bench_torch_backends.py --model Qwen/Qwen2.5-0.5B-Instruct
    .venv/python evals/bench_torch_backends.py --device cuda --processes 4

Each worker loads the model once, warms up, then times `decide()` over the same
batch of contexts. Output: one JSON line per configuration on stdout.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import sys
import time


FRAUD_SCHEMA = {
    "risk": {"type": "enum", "choices": ["LOW", "MEDIUM", "HIGH"],
             "description": "fraud risk of this transaction"},
    "needs_review": {"type": "boolean",
                     "description": "true if a human must check the transaction"},
    "action": {"type": "enum", "choices": ["APPROVE", "REVIEW", "BLOCK"],
               "description": "recommended handling action"},
}

CONTEXT = (
    "Transaction alert. Card holder: private individual, account age 14 months, "
    "previous chargebacks: 0. Amount 9,800 EUR to a beneficiary in Cyprus added "
    "11 minutes ago. Device fingerprint never seen before, IP from a residential "
    "proxy, login 3 minutes before the transfer at 03:14 local time. The customer "
    "did not respond to the in-app confirmation prompt. Merchant category: "
    "crypto exchange. Velocity: 4 transactions in the last hour, 2 declined by "
    "issuer. Notes from the first-line agent: 'customer email auto-reply says on "
    "vacation until next week'."
) * 6  # ~1k tokens of context, closer to the invoice-shaped workloads


def _worker(model: str, device: str, n: int, queue) -> None:
    from parallel_decisions import Decider, Schema

    decider = Decider(backend="torch", model_id=model,
                      torch_device=device, warmup=True, verbose=False)
    schema = Schema(FRAUD_SCHEMA)
    decider.decide(CONTEXT, schema)              # warmup (compile, allocator)
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = decider.decide(CONTEXT, schema)
        latencies.append((time.perf_counter() - t0) * 1000)
        last = result
    queue.put({
        "pid": _pid(),
        "device": last.telemetry.get("device", device),
        "prompt_tokens": last.telemetry.get("prompt_tokens"),
        "latencies_ms": [round(x, 1) for x in latencies],
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 1),
        "sample": last.json(),
    })


def _pid() -> int:
    import os
    return os.getpid()


def run_serial(model: str, device: str, n: int) -> dict:
    queue = mp.Queue()
    _worker(model, device, n, queue)
    out = queue.get()
    out["mode"] = f"serial/{device}"
    return out


def run_parallel(model: str, device: str, processes: int, n: int) -> dict:
    queue = mp.Queue()
    procs = [mp.Process(target=_worker, args=(model, device, n, queue))
             for _ in range(processes)]
    t0 = time.perf_counter()
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join()
    wall = (time.perf_counter() - t0) * 1000
    all_lat = [x for r in results for x in r["latencies_ms"]]
    return {
        "mode": f"parallel x{processes}/{device}",
        "device": results[0]["device"],
        "prompt_tokens": results[0]["prompt_tokens"],
        "latencies_ms": [round(x, 1) for x in all_lat],
        "p50_ms": round(statistics.median(all_lat), 1),
        "p95_ms": round(sorted(all_lat)[int(0.95 * len(all_lat)) - 1], 1),
        "wall_ms": round(wall, 1),
        "throughput_rps": round(len(all_lat) / (wall / 1000), 2),
        "sample": results[0]["sample"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--runs", type=int, default=20, help="decisions per worker")
    parser.add_argument("--processes", type=int, default=0,
                        help="also run N model processes in parallel")
    parser.add_argument("--cpu-baseline", action="store_true",
                        help="also measure the same workload on CPU")
    args = parser.parse_args()

    lines = []
    if args.cpu_baseline:
        lines.append(run_serial(args.model, "cpu", args.runs))
    lines.append(run_serial(args.model, args.device, args.runs))
    if args.processes > 1:
        lines.append(run_parallel(args.model, args.device, args.processes, args.runs))
    for line in lines:
        print(json.dumps(line))
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
