"""openjev command-line interface.

    openjev decide --preset fintech_fraud
    openjev decide --schema FILE --context FILE|-
    openjev decide --json --preset support_triage
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# Presets resolve relative to the repo root (the presets/ dir sits next to openjev/).
_PRESETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "presets")


def load_preset(name: str) -> dict:
    """Load a preset by name ('fintech_fraud'), filename ('fintech_fraud.json'), or path."""
    if os.path.isfile(name):
        path = name
    else:
        filename = name if name.endswith(".json") else f"{name}.json"
        path = os.path.join(_PRESETS_DIR, filename)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_result(preset_title: str, model_id: str, result: dict) -> None:
    """The demo table: preset, latency split, per-field values + confidences."""
    print()
    print(f"Preset : {preset_title}")
    print(f"Model  : {model_id}")
    print(f"Latency: {result['elapsed_ms']:.1f} ms "
          f"(prefill {result['prefill_ms']:.1f} + batched pass {result['suffix_eval_ms']:.1f})")
    print()

    rows = [
        (name, str(entry["value"]), f"{entry['confidence']:.3f}", entry["type"])
        for name, entry in result["field_telemetry"].items()
    ]
    width = max(len(r[0]) for r in rows) if rows else 10

    print(f"{'field':<{width}}  {'value':<22}  conf   type")
    print(f"{'-' * width}  {'-' * 22}  -----  -----")
    for name, value, conf, ftype in rows:
        print(f"{name:<{width}}  {value:<22}  {conf}  {ftype}")

    print()
    print("Assembled JSON (never generated token-by-token, so always well-formed):")
    print(json.dumps(result["parsed_json"], indent=2, default=str))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="openjev", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    decide = sub.add_parser("decide", help="Run parallel constrained decisions on a preset or schema/context")
    decide.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id for mlx-lm")
    decide.add_argument("--preset", help="preset name (presets/NAME.json) or path to a .json file")
    decide.add_argument("--schema", help="path to a schema .json file")
    decide.add_argument("--context", help="path to a context .txt file, or - for stdin")
    decide.add_argument("--json", action="store_true", dest="as_json",
                        help="print the assembled JSON only")
    args = ap.parse_args(argv)

    if args.command == "decide":
        if bool(args.preset) == bool(args.schema or args.context):
            decide.error("use --preset NAME  or  --schema FILE --context FILE|- (not both, not neither)")

        if args.preset:
            preset = load_preset(args.preset)
            schema_dict = preset["schema"]
            context = preset["context"]
            title = preset.get("title", args.preset)
        else:
            with open(args.schema, encoding="utf-8") as f:
                schema_dict = json.load(f)
            if args.context == "-":
                context = sys.stdin.read()
            else:
                with open(args.context, encoding="utf-8") as f:
                    context = f.read()
            title = args.schema

        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)

        schema = StructuredSchema(schema_dict)
        result = run_parallel_generation(model, tokenizer, context, schema)

        if args.as_json:
            print(json.dumps(result["parsed_json"], indent=2, default=str))
        else:
            print_result(title, args.model, result)
