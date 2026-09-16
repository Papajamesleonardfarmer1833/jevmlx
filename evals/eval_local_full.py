#!/usr/bin/env python3
"""Run ALL public TypeSafe eval cases through the local parallel-decisions engine.

Covers the 4 workflows (20 cases / 373 reference pairs).
Output: evals/results/full-local-<tag>.json  in the published_answers shape:
  {"<wf>": {"<case_id>": {qid: {"raw":..., "kind":...}}}}
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


def build_schema(questions: list[dict]) -> dict:
    schema = {}
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
    ap.add_argument("--tag", default="qwen3-8b")
    ap.add_argument("--full", default=os.path.join(HERE, "full_eval.json"))
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--workflow", default=None, help="only this workflow")
    ap.add_argument("--case", default=None, help="only this case_id")
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

    full = json.load(open(args.full))
    out: dict = {}
    timings: dict = {}
    for wf, wdata in full["workflows"].items():
        if args.workflow and wf != args.workflow:
            continue
        out[wf] = {}
        for case in wdata["cases"]:
            cid = case["case_id"]
            if args.case and cid != args.case:
                continue
            schema = StructuredSchema(build_schema(case["questions"]))
            print(f"[{wf}/{cid}] {len(case['questions'])} q, {case['input_chars']} chars ...", flush=True)
            res = run_parallel_generation(case["input_text"], schema)

            answers = {}
            for q in case["questions"]:
                qid = q["qid"]
                entry = res["field_telemetry"][qid]
                if q["type"] == "noul":
                    p_true = next((float(c["probability"]) for c in entry["top_choices"]
                                   if str(c["choice"]).lower() == "true"),
                                  1.0 - float(entry["confidence"]))
                    answers[qid] = {"raw": p_true, "kind": "noul"}
                else:
                    answers[qid] = {"raw": entry["value"], "kind": q["type"]}
            out[wf][cid] = answers
            timings[f"{wf}/{cid}"] = {
                "elapsed_ms": res["elapsed_ms"],
                "prefill_ms": res["prefill_ms"],
                "suffix_eval_ms": res["suffix_eval_ms"],
                "fields": len(case["questions"]),
            }
            print(f"    {res['elapsed_ms']:.0f} ms "
                  f"(prefill {res['prefill_ms']:.0f} + pass {res['suffix_eval_ms']:.0f})", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"full-local-{args.tag}.json")
    json.dump({"model": args.model, "timings": timings, "answers": out}, open(path, "w"), indent=1)
    n = sum(len(v) for wf in out.values() for v in wf.values())
    print(f"saved {path} ({n} answers)")


if __name__ == "__main__":
    main()
