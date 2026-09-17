#!/usr/bin/env python3
"""Run the labeled quality eval against one mlx-lm model using the openjev engine.

Usage:
    .venv/bin/python quality-eval/run_eval.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
    .venv/bin/python quality-eval/run_eval.py --model <hf-id> --limit 4 --tag smoke   # smoke test

Loads exactly one model, runs all cases from cases.json (24 by default), writes
results/<tag>.json with per-case predictions, per-field correctness, confidences
and timings. One process per model; run models sequentially.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
def tag_for(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Labeled quality eval for the parallel decision engine.")
    ap.add_argument("--model", required=True, help="Hugging Face model id served by mlx-lm")
    ap.add_argument("--cases", default=os.path.join(HERE, "cases.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--tag", default="", help="output tag; defaults to the model name")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cases (smoke tests)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    tag = args.tag or tag_for(args.model)
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{tag}.json")

    with open(args.cases) as fh:
        case_file = json.load(fh)

    order = []  # (family_key, case)
    for family_key, fam in case_file["families"].items():
        for case in fam["cases"]:
            order.append((family_key, case))
    if args.limit and args.limit > 0:
        order = order[:args.limit]

    from openjev.engine import load_engine, run_parallel_generation  # noqa: E402
    from openjev.schema import StructuredSchema  # noqa: E402

    print(f"model: {args.model}", flush=True)
    print(f"cases: {len(order)} from {args.cases}", flush=True)

    t0 = time.perf_counter()
    try:
        _, tokenizer = load_engine(args.model)
    except Exception as exc:  # keep run_all.sh going when a model cannot be loaded
        failure = {
            "kind": "rlcd-quality-eval",
            "status": "load_failed",
            "model": args.model,
            "tag": tag,
            "error": f"{type(exc).__name__}: {exc}",
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        with open(out_path, "w") as fh:
            json.dump(failure, fh, indent=2)
        print(f"ERROR: model failed to load: {exc}")
        print(f"wrote {out_path} with status=load_failed")
        return 2
    load_seconds = time.perf_counter() - t0
    print(f"loaded in {load_seconds:.1f}s", flush=True)

    schemas = {key: StructuredSchema(fam["schema"]) for key, fam in case_file["families"].items()}

    results = []
    for family_key, case in order:
        schema = schemas[family_key]
        entry = {
            "id": case["id"],
            "family": family_key,
            "primary_field": case["primary_field"],
            "labels": case["labels"],
            "acceptable": case["acceptable"],
            "note": case["note"],
        }
        try:
            out = run_parallel_generation(case["context"], schema)
        except Exception as exc:
            entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            results.append(entry)
            print(f"  {case['id']}: ERROR {exc}", flush=True)
            continue

        predicted = {name: out["parsed_json"][name]["value"] for name in case["labels"]}
        confidence = {name: out["parsed_json"][name]["prob"] for name in case["labels"]}
        top = {
            name: [[c["choice"], c["probability"]] for c in out["field_telemetry"][name]["top_choices"][:3]]
            for name in case["labels"]
        }
        field_correct, field_acceptable, on_alt = {}, {}, []
        for name, label in case["labels"].items():
            hit = predicted[name] == label
            alt = case["acceptable"].get(name, [])
            field_correct[name] = hit
            field_acceptable[name] = hit or predicted[name] in alt
            if not hit and predicted[name] in alt:
                on_alt.append(name)

        entry.update({
            "status": "ok",
            "context_tokens": len(tokenizer.encode(case["context"])),
            "elapsed_ms": out["elapsed_ms"],
            "prefill_ms": out["prefill_ms"],
            "suffix_eval_ms": out["suffix_eval_ms"],
            "predicted": predicted,
            "confidence": confidence,
            "top_choices": top,
            "field_correct": field_correct,
            "field_correct_or_acceptable": field_acceptable,
            "fields_on_acceptable_alternative": on_alt,
            "all_fields_correct": all(field_correct.values()),
            "all_fields_correct_or_acceptable": all(field_acceptable.values()),
        })
        results.append(entry)

        wrong = [f"{k}={entry['predicted'][k]}(want {entry['labels'][k]})"
                 for k in case["labels"] if not field_correct[k]]
        alt = f" on-alternative:{','.join(on_alt)}" if on_alt else ""
        print(f"  {case['id']}: {'OK ' if not wrong else 'MISS'} "
              f"{entry['elapsed_ms']:.0f}ms conf={min(confidence.values()):.2f}-{max(confidence.values()):.2f}"
              f"{alt}{' ' + '; '.join(wrong) if wrong else ''}", flush=True)

    ok_cases = [r for r in results if r.get("status") == "ok"]
    payload = {
        "kind": "rlcd-quality-eval",
        "status": "ok" if len(ok_cases) == len(order) else "partial",
        "model": args.model,
        "tag": tag,
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "cases_file": os.path.abspath(args.cases),
        "load_seconds": round(load_seconds, 1),
        "case_count": len(results),
        "ok_count": len(ok_cases),
        "total_elapsed_ms": round(sum(r.get("elapsed_ms", 0.0) for r in ok_cases), 1),
        "results": results,
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
