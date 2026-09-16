#!/usr/bin/env python3
"""Benchmark a local mlx-lm model with the parallel-constrained-decoding engine.

Compares, on the same presets:
  - naive autoregressive JSON generation (baseline)
  - parallel constrained decisions (one batched pass over all schema fields)

Usage:
    .venv/bin/python tools/bench_model.py mlx-community/Qwen2.5-1.5B-Instruct-4bit
    .venv/bin/python tools/bench_model.py mlx-community/Qwen2.5-7B-Instruct-4bit --tag 7b

The upstream engine is cloned into vendor/Qwen-2.5-1B-RLCD by setup.sh.
Results are written to results/<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARTIFACT = os.path.join(ROOT, "vendor", "Qwen-2.5-1B-RLCD")
DEFAULT_PRESETS = [
    "fintech_fraud.json",
    "support_triage.json",
    "high_cardinality_255.json",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_id", help="Hugging Face model id (mlx-lm compatible)")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT, help="path to the cloned upstream engine")
    ap.add_argument("--presets", nargs="+", default=DEFAULT_PRESETS)
    ap.add_argument("--tag", default=None, help="output name (default: model id with '/' -> '_')")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    artifact = os.path.abspath(args.artifact)
    if not os.path.isdir(artifact):
        sys.exit(f"engine not found at {artifact} - run ./setup.sh first")
    sys.path.insert(0, artifact)

    import core.engine_mlx as engine_mlx  # noqa: E402

    engine_mlx.MODEL_ID = args.model_id
    print(f"MODEL_ID -> {engine_mlx.MODEL_ID}", flush=True)

    from core.engine_mlx import (  # noqa: E402
        get_engine,
        run_naive_generation,
        run_parallel_generation,
    )
    from core.schema import StructuredSchema  # noqa: E402

    t0 = time.perf_counter()
    get_engine()  # loads the model + Metal warmup
    load_s = round(time.perf_counter() - t0, 1)
    print(f"[load+warmup] {load_s}s", flush=True)

    rows = []
    for rel in args.presets:
        with open(os.path.join(artifact, "presets", rel), encoding="utf-8") as f:
            preset = json.load(f)
        schema = StructuredSchema(preset["schema"])
        print(f"--> {preset['title']}", flush=True)

        naive = run_naive_generation(preset["context"], schema)
        parallel = run_parallel_generation(preset["context"], schema)
        speedup = naive["elapsed_ms"] / max(parallel["elapsed_ms"], 1.0)

        row = {
            "preset": preset["id"],
            "title": preset["title"],
            "num_fields": len(preset["schema"]),
            "naive_ms": naive["elapsed_ms"],
            "naive_tokens": naive["total_tokens"],
            "naive_tokens_per_s": naive["tokens_per_second"],
            "naive_schema_match": naive["schema_match"],
            "parallel_ms": parallel["elapsed_ms"],
            "parallel_prefill_ms": parallel["prefill_ms"],
            "parallel_suffix_ms": parallel["suffix_eval_ms"],
            "parallel_schema_match": parallel["schema_match"],
            "speedup": round(speedup, 1),
            "confidences": {k: v["confidence"] for k, v in parallel["field_telemetry"].items()},
        }
        rows.append(row)

        print(
            f"    naive    {naive['elapsed_ms']:.0f} ms | {naive['total_tokens']} tok "
            f"| schema_ok={naive['schema_match']}",
            flush=True,
        )
        print(
            f"    parallel {parallel['elapsed_ms']:.0f} ms "
            f"(prefill {parallel['prefill_ms']:.0f} + batched pass {parallel['suffix_eval_ms']:.0f}) "
            f"| schema_ok={parallel['schema_match']} | speedup {speedup:.1f}x",
            flush=True,
        )

    tag = args.tag or args.model_id.replace("/", "_")
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model_id, "load_seconds": load_s, "results": rows}, f, indent=2)
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
