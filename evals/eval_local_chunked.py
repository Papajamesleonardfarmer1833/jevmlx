#!/usr/bin/env python3
"""Chunked local runner: same as eval_local_full but splits fields into chunks.

The engine broadcasts the KV cache once per schema field, so memory scales with
(fields x context). Invoice cases have 48 fields over ~8.5k-token inputs -> tens of
GB if run in one pass. Chunking trades a few extra prefills for a bounded footprint.

Usage:
  .venv/bin/python evals/eval_local_chunked.py --model mlx-community/Qwen3-8B-4bit \
      --tag qwen3-8b --chunk-size 6 --workflow invoice_processing
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
    schema = {}
    for q in questions:
        crit = q["criteria"]
        if q["type"] == "noul":
            if isinstance(crit, dict):
                desc = f"{q['instructions']} (true: {crit.get('true','')}; false: {crit.get('false','')})"
            else:
                desc = q["instructions"]
            schema[q["qid"]] = {"type": "boolean", "description": desc}
        elif q["type"] == "score":
            levels = "; ".join(f"{i}={v}" for i, v in enumerate(crit)) if isinstance(crit, list) else ""
            schema[q["qid"]] = {"type": "enum", "choices": ["0", "1", "2", "3"],
                                "description": f"{q['instructions']} Levels: {levels}"}
        else:
            if isinstance(crit, dict):
                opts = "; ".join(f"{k}={v}" for k, v in crit.items())
                choices = list(crit.keys())
            else:
                opts, choices = "", []
            schema[q["qid"]] = {"type": "enum", "choices": choices,
                                "description": f"{q['instructions']} Options: {opts}"}
    return schema


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    ap.add_argument("--tag", default="qwen3-8b")
    ap.add_argument("--chunk-size", type=int, default=6)
    ap.add_argument("--full", default=os.path.join(HERE, "full_eval.json"))
    ap.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--workflow", default=None)
    ap.add_argument("--case", default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.abspath(args.artifact))
    import core.engine_mlx as engine_mlx  # noqa: E402

    engine_mlx.MODEL_ID = args.model
    print(f"MODEL_ID -> {engine_mlx.MODEL_ID} (chunk={args.chunk_size})", flush=True)

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
        out.setdefault(wf, {})
        for case in wdata["cases"]:
            cid = case["case_id"]
            if args.case and cid != args.case:
                continue
            questions = case["questions"]
            chunks = [questions[i:i + args.chunk_size] for i in range(0, len(questions), args.chunk_size)]
            print(f"[{wf}/{cid}] {len(questions)} q in {len(chunks)} chunk(s), "
                  f"{case['input_chars']} chars ...", flush=True)

            answers: dict = {}
            tot_ms = tot_pre = tot_suf = 0.0
            for ci, chunk in enumerate(chunks, 1):
                schema = StructuredSchema(build_schema_dict(chunk))
                t1 = time.perf_counter()
                res = run_parallel_generation(case["input_text"], schema)
                dt = (time.perf_counter() - t1) * 1000
                tot_ms += dt
                tot_pre += res["prefill_ms"]
                tot_suf += res["suffix_eval_ms"]
                for q in chunk:
                    qid = q["qid"]
                    entry = res["field_telemetry"][qid]
                    if q["type"] == "noul":
                        p_true = next((float(c["probability"]) for c in entry["top_choices"]
                                       if str(c["choice"]).lower() == "true"),
                                      1.0 - float(entry["confidence"]))
                        answers[qid] = {"raw": p_true, "kind": "noul"}
                    else:
                        answers[qid] = {"raw": entry["value"], "kind": q["type"]}
                print(f"    chunk {ci}/{len(chunks)}: {dt:.0f} ms "
                      f"(prefill {res['prefill_ms']:.0f} + pass {res['suffix_eval_ms']:.0f})", flush=True)

            out[wf][cid] = answers
            timings[f"{wf}/{cid}"] = {
                "total_ms": tot_ms, "prefill_ms": tot_pre, "suffix_eval_ms": tot_suf,
                "chunks": len(chunks), "fields": len(questions),
            }
            print(f"    -> case total {tot_ms:.0f} ms, {len(answers)} answers", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"full-local-{args.tag}.json")
    existing = {}
    if os.path.isfile(path):
        existing = json.load(open(path))
    merged_answers = existing.get("answers", {})
    for wf, cases in out.items():
        merged_answers.setdefault(wf, {}).update(cases)
    merged_timings = existing.get("timings", {})
    merged_timings.update(timings)
    json.dump({"model": args.model, "timings": merged_timings, "answers": merged_answers},
              open(path, "w"), indent=1)
    n = sum(len(v) for wf in merged_answers.values() for v in wf.values())
    print(f"saved {path} ({n} answers total)")


if __name__ == "__main__":
    main()
