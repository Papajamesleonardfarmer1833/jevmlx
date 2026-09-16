#!/usr/bin/env python3
"""Run TypeSafe's security-incident questions through the LOCAL parallel-decisions engine.

Maps the question catalog onto the engine's schema types:
  noul   -> boolean field          (answer: P(true) from the candidate distribution)
  score  -> enum ["0","1","2","3"] (legend included in the field description)
  choice -> enum of criteria keys  (options included in the description,
                                    because the engine renders only the description in its prompt)

Only the questions that the reference answers cover are asked per case (3 or 14).

Usage:
  .venv/bin/python evals/eval_local.py --model mlx-community/Qwen3-8B-4bit --tag local-qwen3-8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_ARTIFACT = os.path.join(ROOT, "source", "Qwen-2.5-1B-RLCD")


def build_schema_dict(questions: list[dict]) -> dict:
    schema: dict = {}
    for q in questions:
        crit = q["criteria"]
        if q["type"] == "noul":
            desc = f"{q['instructions']} (true: {crit['true']}; false: {crit['false']})"
            schema[q["qid"]] = {"type": "boolean", "description": desc}
        elif q["type"] == "score":
            levels = "; ".join(f"{i}={v}" for i, v in enumerate(crit))
            schema[q["qid"]] = {"type": "enum", "choices": ["0", "1", "2", "3"],
                                "description": f"{q['instructions']} Levels: {levels}"}
        else:
            opts = "; ".join(f"{k}={v}" for k, v in crit.items())
            schema[q["qid"]] = {"type": "enum", "choices": list(crit.keys()),
                                "description": f"{q['instructions']} Options: {opts}"}
    return schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--tag", default="local-qwen3-8b")
    ap.add_argument("--mini", default=os.path.join(HERE, "typesafe", "mini_eval.json"))
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    args = ap.parse_args()

    artifact = os.path.abspath(args.artifact)
    sys.path.insert(0, artifact)

    import core.engine_mlx as engine_mlx  # noqa: E402

    engine_mlx.MODEL_ID = args.model
    print(f"MODEL_ID -> {engine_mlx.MODEL_ID}", flush=True)

    from core.engine_mlx import get_engine, run_parallel_generation  # noqa: E402
    from core.schema import StructuredSchema  # noqa: E402

    t0 = time.perf_counter()
    get_engine()
    print(f"[load+warmup] {time.perf_counter() - t0:.1f}s", flush=True)

    mini = json.load(open(args.mini))
    qtype = {q["qid"]: q["type"] for q in mini["questions"]}

    results: dict[str, dict] = {}
    timings: dict[str, dict] = {}
    for case in mini["cases"]:
        relevant = [q for q in mini["questions"] if q["qid"] in case["reference"]]
        schema = StructuredSchema(build_schema_dict(relevant))
        doc = case["docs"][0]
        context = doc.get("alert", "").strip() + "\n\n" + "\n".join(
            str(r) for r in doc.get("context", {}).get("records", [])
        )

        print(f"[{case['case_id']}] {len(relevant)} questions ...", flush=True)
        out = run_parallel_generation(context, schema)

        answers = {}
        for q in relevant:
            qid = q["qid"]
            entry = out["field_telemetry"][qid]
            if q["type"] == "noul":
                prob_true = next(
                    (float(c["probability"]) for c in entry["top_choices"]
                     if str(c["choice"]).lower() == "true"),
                    1.0 - float(entry["confidence"]),
                )
                answers[qid] = {"value": prob_true >= 0.5, "raw": prob_true, "kind": "noul"}
            elif q["type"] == "score":
                answers[qid] = {"value": str(entry["value"]), "raw": entry["value"], "kind": "score"}
            else:
                answers[qid] = {"value": str(entry["value"]), "raw": entry["value"], "kind": "choice"}
        results[case["case_id"]] = answers
        timings[case["case_id"]] = {
            "elapsed_ms": out["elapsed_ms"],
            "prefill_ms": out["prefill_ms"],
            "suffix_eval_ms": out["suffix_eval_ms"],
            "fields": len(relevant),
        }
        print(f"    {out['elapsed_ms']:.0f} ms (prefill {out['prefill_ms']:.0f} + pass {out['suffix_eval_ms']:.0f})",
              flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    payload = {"model": args.model, "timings": timings, "answers": results}
    path = os.path.join(args.outdir, f"{args.tag}.json")
    json.dump(payload, open(path, "w"), indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
