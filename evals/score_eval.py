#!/usr/bin/env python3
"""Score every model's answers against the TypeSafe consensus reference.

Models compared on the same 5 cases x question catalog:
  - opus / sol / typesafe (Jev): from the published viewer data (mini_eval.json)
  - deepseek-v4.1-flash: evals/results/deepseek-v4.1-flash.json
  - local-qwen3-8b: evals/results/local-qwen3-8b.json

Metric: agreement with the consensus argmax (mean of GPT-6 Astra and Fable 5.1
probabilities, as shipped in the viewer data) for every question the reference covers.

Usage:
  .venv/bin/python evals/score_eval.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MINI = os.path.join(HERE, "typesafe", "mini_eval.json")
RESULTS = os.path.join(HERE, "results")

EXTRA_MODELS = {
    "deepseek-v4.1-flash": os.path.join(RESULTS, "deepseek-v4.1-flash.json"),
    "local-qwen3-8b": os.path.join(RESULTS, "local-qwen3-8b.json"),
}


def consensus(ref_entry: dict) -> str | None:
    probs = ref_entry.get("probs") or {}
    if not probs:
        return None
    return max(probs, key=probs.get)


def canonical(model_value, kind: str, qtype: str) -> str | None:
    """Map any model's answer to a comparable canonical string."""
    if qtype == "noul":
        if isinstance(model_value, bool):
            return str(model_value).lower()
        if isinstance(model_value, (int, float)):
            return str(model_value >= 0.5).lower()
        if isinstance(model_value, str):
            return model_value.lower()
        return None
    if qtype == "score":
        try:
            lvl = int(round(float(model_value)))
        except (TypeError, ValueError):
            return None
        return str(max(0, min(3, lvl)))
    return str(model_value)


def main() -> None:
    mini = json.load(open(MINI))
    qtype = {q["qid"]: q["type"] for q in mini["questions"]}

    # model -> {case_id: {qid: value}} plus kinds for extra models
    models: dict[str, dict[str, dict[str, str]]] = {}

    for mkey in ("opus", "sol", "typesafe"):
        models[mkey] = {}
        for case in mini["cases"]:
            answers = case["model_answers"].get(mkey, {})
            models[mkey][case["case_id"]] = {
                qid: canonical(a["value"], "mini", qtype[qid]) for qid, a in answers.items()
            }

    for name, path in EXTRA_MODELS.items():
        if not os.path.isfile(path):
            print(f"(skipping {name}: {path} not found)")
            continue
        data = json.load(open(path))
        models[name] = {
            case_id: {
                qid: canonical(a["value"], a.get("kind", ""), qtype[qid])
                for qid, a in answers.items()
            }
            for case_id, answers in data["answers"].items()
        }

    # score
    rows = []
    per_type: dict[str, dict[str, list[bool]]] = {}
    for name, by_case in models.items():
        ok = total = 0
        for case in mini["cases"]:
            ref = case["reference"]
            answers = by_case.get(case["case_id"], {})
            for qid, ref_entry in ref.items():
                want = consensus(ref_entry)
                got = answers.get(qid)
                if want is None or got is None:
                    continue
                hit = got == want
                ok += hit
                total += 1
                per_type.setdefault(name, {}).setdefault(qtype[qid], []).append(hit)
        rows.append((name, ok, total))

    # --- strict common subset: only (case, qid) pairs that EVERY model answered ---
    common_pairs = []
    for case in mini["cases"]:
        for qid, ref_entry in case["reference"].items():
            if consensus(ref_entry) is None:
                continue
            if all(qid in by_case.get(case["case_id"], {}) for by_case in models.values()):
                common_pairs.append((case["case_id"], qid))

    common_rows = []
    for name, by_case in models.items():
        ok = 0
        for case_id, qid in common_pairs:
            case = next(c for c in mini["cases"] if c["case_id"] == case_id)
            want = consensus(case["reference"][qid])
            got = by_case[case_id][qid]
            ok += got == want
        common_rows.append((name, ok, len(common_pairs)))

    width = max(len(n) for n, _, _ in rows) + 2
    print()
    print(f"{'model':<{width}} {'agree':>5} {'n':>3} {'acc':>7}   per-type (noul/score/choice)")
    print("-" * (width + 42))
    for name, ok, total in sorted(rows, key=lambda r: -(r[1] / r[2] if r[2] else 0)):
        acc = f"{ok / total * 100:.1f}%" if total else "n/a"
        parts = []
        for t in ("noul", "score", "choice"):
            hits = per_type.get(name, {}).get(t, [])
            parts.append(f"{sum(hits)}/{len(hits)}" if hits else "-")
        print(f"{name:<{width}} {ok:>5} {total:>3} {acc:>7}   {parts[0]} / {parts[1]} / {parts[2]}")

    print()
    print("Strict common subset (only questions answered by EVERY model listed):")
    print(f"{'model':<{width}} {'agree':>5} {'n':>3} {'acc':>7}")
    print("-" * (width + 22))
    for name, ok, total in sorted(common_rows, key=lambda r: -(r[1] / r[2] if r[2] else 0)):
        acc = f"{ok / total * 100:.1f}%" if total else "n/a"
        print(f"{name:<{width}} {ok:>5} {total:>3} {acc:>7}")
    print(f"(overlap pairs: {len(common_pairs)})")

    print()
    print("Note: these 5 cases are the viewer's curated *disagreement* examples, so absolute")
    print("accuracy here is not comparable to the published per-workflow numbers (61.7%-76.0%).")
    print("Within-table comparisons are apples-to-apples: same cases, same questions, same reference.")

    out = {
        "rows": [{"model": n, "agreements": ok, "n": total} for n, ok, total in rows],
        "common_subset": {
            "n_pairs": len(common_pairs),
            "rows": [{"model": n, "agreements": ok, "n": total} for n, ok, total in common_rows],
        },
        "per_type": {m: {t: [bool(x) for x in v] for t, v in d.items()} for m, d in per_type.items()},
    }
    path = os.path.join(RESULTS, "summary.json")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
