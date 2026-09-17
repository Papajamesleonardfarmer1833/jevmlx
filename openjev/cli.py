"""openjev command-line interface.

openjev decide --preset fintech_fraud
openjev decide --schema FILE --context FILE|-
openjev decide --json --preset support_triage
"""

from __future__ import annotations

import argparse
import copy
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


def _fmt_confidence(value: float, decimals: int) -> str:
    """Presentation rounding for confidences: 3 decimals in the table, 4 in --json.
    The engine returns full-precision floats; rounding happens only here."""
    return f"{value:.{decimals}f}"


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
        (name, str(entry["value"]), _fmt_confidence(entry["confidence"], 3), entry["type"])
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


def _rounded_json_payload(result: dict) -> dict:
    """--json output: engine values with confidences rounded to 4 decimals."""
    parsed = copy.deepcopy(result["parsed_json"])
    for field in parsed.values():
        if isinstance(field, dict) and "prob" in field:
            field["prob"] = round(field["prob"], 4)
    return parsed


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

    eval_p = sub.add_parser(
        "eval",
        help="Run labeled cases through a decision track; writes predictions + run manifest",
    )
    eval_p.add_argument("--data", required=True, help="cases JSONL (see the eval contract)")
    eval_p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id for mlx-lm")
    eval_p.add_argument(
        "--track",
        required=True,
        choices=["parallel", "naive_local", "api_baseline"],
        help="decision track to run",
    )
    eval_p.add_argument("--api-base", help="chat-completions base URL (api_baseline track)")
    eval_p.add_argument("--api-model", help="model name sent to the API (api_baseline track)")
    eval_p.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="env var holding the API key (api_baseline track)",
    )
    eval_p.add_argument(
        "--permutations",
        default="none",
        choices=["none", "rotations", "fieldperm", "all"],
        help="order-sensitivity probe (parallel track only)",
    )
    eval_p.add_argument("--limit", type=int, default=None, help="only the first N cases")
    eval_p.add_argument(
        "--split",
        default="all",
        choices=["train", "holdout", "all"],
        help="case split to run",
    )
    eval_p.add_argument(
        "--out", required=True, help="output directory (predictions.jsonl, run.json)"
    )
    report_p = sub.add_parser(
        "report",
        help="Build a JSON + markdown eval report from predictions.jsonl (offline)",
    )
    report_p.add_argument("--predictions", required=True, help="path to predictions.jsonl")
    report_p.add_argument("--out", required=True, help="output path for the JSON report")
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
            print(json.dumps(_rounded_json_payload(result), indent=2))
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

    elif args.command == "report":
        # Offline: pure-python metrics over predictions.jsonl -> evalreport.
        from openjev.evalmetrics import compute_metrics, load_predictions
        from openjev.evalreport import environment, write_report

        records = load_predictions(args.predictions)
        write_report(args.out, {"environment": environment(), "metrics": compute_metrics(records)})
        print(f"wrote {args.out} (+ .md)")

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
        # compile_error findings mean the schema cannot run at all: exit 1.
        # (collisions only slow the engine down; they stay exit 0 with a
        # warning printed above).
        if any(finding.kind == "compile_error" for finding in findings):
            sys.exit(1)

    elif args.command == "eval":
        _run_eval_command(args)


def _run_eval_command(args) -> None:
    """openjev eval: batch labeled cases through one decision track."""
    import logging

    from openjev import evalrun

    cases = evalrun.load_cases(args.data)
    if args.limit is not None:
        cases = cases[: args.limit]

    extra: dict = {"dataset_path": os.path.abspath(args.data)}
    lock = os.path.join(os.path.dirname(os.path.abspath(args.data)), "dataset.lock.json")

    if args.track == "parallel":
        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)
        decide_fn = evalrun.parallel_decide_fn(model, tokenizer)
        chat_template = getattr(tokenizer, "chat_template", None)
        plan_provider = lambda schema: schema.compile_batch_plan(tokenizer)  # noqa: E731
    elif args.track == "naive_local":
        print(f"Loading {args.model} ...", flush=True)
        model, tokenizer = load_engine(args.model)
        decide_fn = evalrun.naive_local_decide_fn(model, tokenizer)
        chat_template = getattr(tokenizer, "chat_template", None)
        plan_provider = None
    else:
        if not (args.api_base and args.api_model):
            eval_p_error = "api_baseline track requires --api-base and --api-model"
            raise SystemExit(eval_p_error)
        api_key = os.environ.get(args.api_key_env)
        decide_fn, api_params = evalrun.api_baseline_decide_fn(
            args.api_base, args.api_model, api_key
        )
        extra.update(api_params)
        chat_template = None
        plan_provider = None

    logging.getLogger("openjev.evalrun").setLevel(logging.INFO)
    run = evalrun.run_eval(
        cases,
        decide_fn,
        track=args.track,
        model=args.api_model if args.track == "api_baseline" else args.model,
        permutations=args.permutations,
        split=args.split,
        out_dir=args.out,
        extra_config=extra,
        chat_template=chat_template,
        plan_provider=plan_provider,
        dataset_lock_path=lock if os.path.exists(lock) else None,
        dataset_path=os.path.abspath(args.data),
    )
    print(
        f"run {run['run_id']}: {run['counts']['cases']} cases, "
        f"{run['counts']['prediction_lines']} prediction lines -> {args.out}/"
    )
