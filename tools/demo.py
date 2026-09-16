#!/usr/bin/env python3
"""Run a single parallel-decision call on a preset and print the typed result.

Usage:
    .venv/bin/python tools/demo.py                                    # fintech preset, 1.5B
    .venv/bin/python tools/demo.py support_triage
    .venv/bin/python tools/demo.py fintech_fraud --model mlx-community/Qwen2.5-7B-Instruct-4bit

This is the "what does a Jev-style typed decision actually look like" example.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ARTIFACT = os.path.join(ROOT, "vendor", "Qwen-2.5-1B-RLCD")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", default="fintech_fraud", help="preset name or path to a .json file")
    ap.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    args = ap.parse_args()

    artifact = os.path.abspath(args.artifact)
    if not os.path.isdir(artifact):
        sys.exit(f"engine not found at {artifact} - run ./setup.sh first")
    sys.path.insert(0, artifact)

    import core.engine_mlx as engine_mlx  # noqa: E402

    engine_mlx.MODEL_ID = args.model

    from core.engine_mlx import get_engine, run_parallel_generation  # noqa: E402
    from core.schema import StructuredSchema  # noqa: E402

    preset_path = args.preset
    if not preset_path.endswith(".json"):
        preset_path = os.path.join(artifact, "presets", f"{preset_path}.json")
    with open(preset_path, encoding="utf-8") as f:
        preset = json.load(f)

    print(f"Loading {args.model} ...", flush=True)
    get_engine()

    schema = StructuredSchema(preset["schema"])
    result = run_parallel_generation(preset["context"], schema)

    print()
    print(f"Preset : {preset['title']}")
    print(f"Model  : {args.model}")
    print(f"Latency: {result['elapsed_ms']:.1f} ms "
          f"(prefill {result['prefill_ms']:.1f} + batched pass {result['suffix_eval_ms']:.1f})")
    print()

    rows = []
    for name, entry in result["field_telemetry"].items():
        rows.append((name, str(entry["value"]), f"{entry['confidence']:.3f}", entry["type"]))
    width = max(len(r[0]) for r in rows) if rows else 10

    print(f"{'field':<{width}}  {'value':<22}  conf   type")
    print(f"{'-' * width}  {'-' * 22}  -----  -----")
    for name, value, conf, ftype in rows:
        print(f"{name:<{width}}  {value:<22}  {conf}  {ftype}")

    print()
    print("Assembled JSON (never generated token-by-token, so always well-formed):")
    print(json.dumps({k: v for k, v in result["parsed_json"].items()}, indent=2, default=str))


if __name__ == "__main__":
    main()
