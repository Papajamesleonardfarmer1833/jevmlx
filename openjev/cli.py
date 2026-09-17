"""openjev command-line interface.

openjev decide --preset fintech_fraud
openjev decide --schema FILE --context FILE|-
openjev decide --json --preset support_triage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from importlib import resources

from openjev import __version__
from openjev.api import DEFAULT_MODEL
from openjev.engine import load_engine, run_parallel_generation
from openjev.lint import lint_schema
from openjev.log import configure
from openjev.schema import StructuredSchema


def load_preset(name: str) -> dict:
    """Load a preset: a filesystem path if it exists, else a bundled preset
    by name ('fintech_fraud' or 'fintech_fraud.json')."""
    if os.path.isfile(name):
        with open(name, encoding="utf-8") as f:
            return json.load(f)
    filename = name if name.endswith(".json") else f"{name}.json"
    return json.loads(
        resources.files("openjev.presets").joinpath(filename).read_text(encoding="utf-8")
    )


def print_result(preset_title: str, model_id: str, result: dict) -> None:
    """The demo table: preset, latency split, per-field values + confidences."""
    print()
    print(f"Preset : {preset_title}")
    print(f"Model  : {model_id}")
    print(
        f"Latency: {result['elapsed_ms']:.1f} ms "
        f"(prefill {result['prefill_ms']:.1f} + batched pass {result['suffix_eval_ms']:.1f})"
    )
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
    ap.add_argument("--version", action="version", version=f"openjev {__version__}")
    sub = ap.add_subparsers(dest="command", required=True)

    decide = sub.add_parser(
        "decide", help="Run parallel constrained decisions on a preset or schema/context"
    )
    decide.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id for mlx-lm")
    decide.add_argument(
        "--preset",
        help="preset name (bundled: fintech_fraud, support_triage, ...) or path to a .json file",
    )
    decide.add_argument("--schema", help="path to a schema .json file")
    decide.add_argument("--context", help="path to a context .txt file, or - for stdin")
    decide.add_argument(
        "--json", action="store_true", dest="as_json", help="print the assembled JSON only"
    )
    decide.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="softmax temperature for choice probabilities (1.0 = raw)",
    )

    calib = sub.add_parser(
        "calibrate", help="Fit a temperature on labeled JSONL cases and report ECE"
    )
    calib.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id for mlx-lm")
    calib.add_argument(
        "--data", required=True, help="JSONL file: {schema, context, labels} per line"
    )
    calib.add_argument("--bins", type=int, default=10, help="ECE bin count")

    serve_p = sub.add_parser("serve", help="Serve decisions over HTTP (one Metal GPU, serial)")
    serve_p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id for mlx-lm")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    ap.add_argument("-v", "--verbose", action="store_true", help="info-level logs on stderr")

    validate_p = sub.add_parser(
        "validate", help="Lint a schema for engine-visible problems (no model download)"
    )
    validate_p.add_argument("schema", help="path to a schema .json file")
    validate_p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model id whose tokenizer decides choice token boundaries",
    )
    validate_p.add_argument(
        "--json", action="store_true", dest="as_json", help="print findings as JSON"
    )
    args = ap.parse_args(argv)

    # serve defaults to INFO: the user must see the listen address. -v is a no-op there.
    if args.command == "serve":
        level = logging.INFO
    else:
        level = logging.INFO if args.verbose else logging.WARNING
    configure(level=level, json_mode=os.environ.get("OPENJEV_LOG") == "json")

    if args.command == "decide":
        if args.preset and (args.schema or args.context):
            decide.error("--preset cannot be combined with --schema/--context")
        if not args.preset and not (args.schema and args.context):
            missing = [
                flag
                for flag, given in (("--schema", args.schema), ("--context", args.context))
                if not given
            ]
            decide.error(
                "exactly one of --preset or --schema AND --context is required; "
                f"missing: {', '.join(missing)}"
            )

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
        result = run_parallel_generation(
            model, tokenizer, context, schema, temperature=args.temperature
        )

        if args.as_json:
            print(json.dumps(result["parsed_json"], indent=2, default=str))
        else:
            print_result(title, args.model, result)

    elif args.command == "calibrate":
        from openjev import calibrate

        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)
        cases = calibrate.load_cases(args.data)
        print(f"collecting scores from {len(cases)} labeled cases ...", flush=True)
        samples = calibrate.collect(model, tokenizer, cases)

        t_fit = calibrate.fit_temperature(samples)
        ece_before = calibrate.ece(samples, 1.0, bins=args.bins)
        ece_after = calibrate.ece(samples, t_fit, bins=args.bins)
        acc = calibrate.accuracy(samples)
        print(f"n samples      : {len(samples)}")
        print(f"fitted T       : {t_fit}")
        print(f"ECE before     : {ece_before:.4f}  (T=1.0)")
        print(f"ECE after      : {ece_after:.4f}  (T={t_fit})")
        print(f"accuracy       : {acc:.4f}")

    elif args.command == "serve":
        from openjev.serve import serve

        serve(args.model, args.host, args.port)

    elif args.command == "validate":
        from dataclasses import asdict

        # transformers is imported lazily: slow to import and only validate needs it.
        from transformers import AutoTokenizer

        with open(args.schema, encoding="utf-8") as f:
            schema = StructuredSchema(json.load(f))
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        findings = lint_schema(schema, tokenizer)

        if args.as_json:
            print(json.dumps([asdict(finding) for finding in findings], indent=2))
        else:
            if not findings:
                print("OK: no findings.")
            for finding in findings:
                print(f"{finding.field}: [{finding.kind}] {finding.message}")
                if finding.suggestion:
                    print(f"  suggestion: {finding.suggestion}")
        if any(finding.kind == "collision" for finding in findings):
            sys.exit(1)
