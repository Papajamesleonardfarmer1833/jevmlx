#!/usr/bin/env python3
"""Cache-reusing local runner: prefill once per case, evaluate fields in chunks.

Why: the stock engine prefills per call, so chunked runs pay the prefill N times.
Invoice cases are ~8.5k tokens; on a 16 GB M5 that turns into minutes per chunk
under memory pressure. Here the KV cache is built once and reused for each chunk
of fields, which is both faster and closer to the engine's intended behaviour.

Usage:
  .venv/bin/python evals/eval_local_cached.py --model mlx-community/Qwen2.5-7B-Instruct-4bit \
      --tag qwen2.5-7b --chunk-size 8 --workflow invoice_processing
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def build_schema_dict(questions: list[dict], tokenizer, schema_mod):
    """Same mapping as the other runners (boolean / enum)."""
    out = {}
    for q in questions:
        crit = q["criteria"]
        if q["type"] == "noul":
            desc = q["instructions"]
            if isinstance(crit, dict):
                desc += f" (true: {crit.get('true','')}; false: {crit.get('false','')})"
            out[q["qid"]] = {"type": "boolean", "description": desc}
        elif q["type"] == "score":
            levels = "; ".join(f"{i}={v}" for i, v in enumerate(crit)) if isinstance(crit, list) else ""
            out[q["qid"]] = {"type": "enum", "choices": ["0", "1", "2", "3"],
                             "description": f"{q['instructions']} Levels: {levels}"}
        else:
            opts = "; ".join(f"{k}={v}" for k, v in crit.items()) if isinstance(crit, dict) else ""
            out[q["qid"]] = {"type": "enum", "choices": list(crit.keys()) if isinstance(crit, dict) else [],
                             "description": f"{q['instructions']} Options: {opts}"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--full", default=os.path.join(HERE, "full_eval.json"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "results"))
    ap.add_argument("--workflow", default=None)
    ap.add_argument("--case", default=None)
    args = ap.parse_args()

    from openjev.schema import StructuredSchema  # noqa: E402

    print(f"loading {args.model} ...", flush=True)
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"[loaded] {time.perf_counter() - t0:.1f}s", flush=True)

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
                  f"{case['input_chars']} chars", flush=True)

            # ---- prefill once ----
            schema_all = StructuredSchema(build_schema_dict(questions, tokenizer, None))
            base_prompt = (
                "<|im_start|>system\n"
                f"Classify JSON attributes:\n{schema_all.to_parallel_schema_str()}<|im_end|>\n"
                "<|im_start|>user\n"
                f"{case['input_text']}<|im_end|>\n"
                "<|im_start|>assistant\n{{\n"
            )
            base_toks = tokenizer.encode(base_prompt)
            base_arr = mx.array(base_toks)[None]
            t_pre0 = time.perf_counter()
            cache = make_prompt_cache(model)
            model(base_arr, cache=cache)
            mx.eval(*[c.keys for c in cache if hasattr(c, "keys") and c.keys is not None])
            prefill_ms = (time.perf_counter() - t_pre0) * 1000
            print(f"    prefill {prefill_ms:.0f} ms ({len(base_toks)} tokens)", flush=True)

            # ---- per chunk: broadcast from the same cache ----
            answers: dict = {}
            for ci, chunk in enumerate(chunks, 1):
                schema = StructuredSchema(build_schema_dict(chunk, tokenizer, None))
                meta = schema.compile_parallel_metadata(tokenizer)
                suffixes_batch = meta["suffixes_batch"]
                M = suffixes_batch.shape[0]

                b_cache = []
                for c in cache:
                    nc = copy.copy(c)
                    if hasattr(c, "keys") and c.keys is not None:
                        nc.keys = mx.repeat(c.keys, M, axis=0)
                        nc.values = mx.repeat(c.values, M, axis=0)
                    b_cache.append(nc)

                t_s0 = time.perf_counter()
                suffix_out = model(suffixes_batch, cache=b_cache)
                mx.eval(suffix_out)
                suffix_ms = (time.perf_counter() - t_s0) * 1000

                for i, (fname, fdef) in enumerate(meta["field_items"]):
                    decision_idx = meta["suffix_lengths"][i] - 1
                    field_logits = suffix_out[i, decision_idx, :]
                    cand_tokens = meta["cands_per_field"][i]
                    scores = [float(field_logits[tid]) for tid in cand_tokens]
                    scores_arr = mx.array(scores)
                    probs = mx.softmax(scores_arr)
                    mx.eval(probs)
                    w_idx = int(mx.argmax(probs))
                    w_prob = float(probs[w_idx])
                    ftype = fdef.field_type
                    if ftype == "boolean":
                        val = (["true", "false"][w_idx].lower() == "true")
                        answers[fname] = {"raw": w_prob, "kind": "noul"}
                    elif fdef.choices and all(c in ("0", "1", "2", "3") for c in fdef.choices):
                        answers[fname] = {"raw": str(w_idx), "kind": "score"}
                    else:
                        answers[fname] = {"raw": fdef.choices[w_idx], "kind": "choice"}

                print(f"    chunk {ci}/{len(chunks)}: {suffix_ms:.0f} ms (batch {M})", flush=True)

            out[wf][cid] = answers
            timings[f"{wf}/{cid}"] = {"prefill_ms": prefill_ms, "chunks": len(chunks),
                                      "fields": len(questions)}
            print(f"    -> {len(answers)} answers", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"full-local-{args.tag}.json")
    existing = json.load(open(path)) if os.path.isfile(path) else {}
    merged = existing.get("answers", {})
    for wf, cases in out.items():
        merged.setdefault(wf, {}).update(cases)
    merged_t = existing.get("timings", {})
    merged_t.update(timings)
    json.dump({"model": args.model, "timings": merged_t, "answers": merged},
              open(path, "w"), indent=1)
    n = sum(len(v) for wf in merged.values() for v in wf.values())
    print(f"saved {path} ({n} answers total)")


if __name__ == "__main__":
    main()
