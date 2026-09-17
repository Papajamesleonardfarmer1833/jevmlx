#!/usr/bin/env python3
"""Export real decision outputs for the static showcase Space.

Runs each preset through the parallel decision engine and the naive baseline
(same model), and writes a single JSON file with everything a static page
needs: per-field values + confidences + top alternatives, latency split, and
the naive baseline result.

Usage:
    .venv/bin/python tools/export_demo_data.py                       # 1.5B, all presets
    .venv/bin/python tools/export_demo_data.py --model mlx-community/Qwen2.5-7B-Instruct-4bit

Requires `uv pip install -e .` first.
"""
from __future__ import annotations

import argparse
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRESETS = ["fintech_fraud", "support_triage", "code_security", "high_cardinality_255"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--out", default=os.path.join(ROOT, "hf-space-static", "data", "demo.json"))
    args = ap.parse_args()

    from openjev.cli import load_preset
    from openjev.engine import load_engine, run_naive_generation, run_parallel_generation
    from openjev.schema import StructuredSchema

    t0 = time.perf_counter()
    model, tokenizer = load_engine(args.model)
    print(f"[load+warmup] {time.perf_counter() - t0:.1f}s", flush=True)

    payload = {"model": args.model, "generated": time.strftime("%Y-%m-%d"), "presets": []}
    for name in PRESETS:
        preset = load_preset(name)
        schema = StructuredSchema(preset["schema"])
        print(f"--> {preset['title']}", flush=True)

        parallel = run_parallel_generation(model, tokenizer, preset["context"], schema)
        naive = run_naive_generation(model, tokenizer, preset["context"], schema)

        fields = []
        for field_name, entry in parallel["field_telemetry"].items():
            top = [{"choice": str(c["choice"]), "p": round(float(c["probability"]), 4)} for c in entry["top_choices"][:4]]
            fields.append({
                "name": field_name,
                "type": entry["type"],
                "value": str(entry["value"]),
                "confidence": round(float(entry["confidence"]), 4),
                "top": top,
            })

        payload["presets"].append({
            "id": preset["id"],
            "title": preset["title"],
            "description": preset.get("description", ""),
            "context": preset["context"],
            "num_fields": len(preset["schema"]),
            "parallel": {
                "elapsed_ms": parallel["elapsed_ms"],
                "prefill_ms": parallel["prefill_ms"],
                "suffix_eval_ms": parallel["suffix_eval_ms"],
                "schema_match": parallel["schema_match"],
                "fields": fields,
                "json": {k: v["value"] for k, v in parallel["parsed_json"].items()},
            },
            "naive": {
                "elapsed_ms": naive["elapsed_ms"],
                "total_tokens": naive["total_tokens"],
                "tokens_per_second": naive["tokens_per_second"],
                "schema_match": naive["schema_match"],
                "raw_text": naive["raw_text"][:2000],
            },
        })
        print(f"    parallel {parallel['elapsed_ms']:.0f} ms | naive {naive['elapsed_ms']:.0f} ms", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"saved {args.out} ({os.path.getsize(args.out) / 1024:.0f} KB)", flush=True)


if __name__ == "__main__":
    main()
