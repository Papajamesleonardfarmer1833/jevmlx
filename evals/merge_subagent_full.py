#!/usr/bin/env python3
"""Merge subagent answer files into one normalized result file.

Input:  evals/subagent_full/<label>.json   -- raw JSON object returned by the model:
          {"<case_id>": {"<qid>": <answer>, ...}, ...}
        Label is one of the 4 workflow names, or an invoice case id (ap_*).
Output: evals/results/full-deepseek-v4.1-flash.json   (published_answers shape)
"""

from __future__ import annotations

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "subagent_full")
FULL = os.path.join(HERE, "full_eval.json")
OUT = os.path.join(HERE, "results", "full-deepseek-v4.1-flash.json")

WORKFLOWS = [
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
]


def main() -> None:
    full = json.load(open(FULL))
    # qid -> type for each workflow
    qtype: dict[str, dict[str, str]] = {}
    case_wf: dict[str, str] = {}
    for wf, wdata in full["workflows"].items():
        qtype[wf] = {}
        for case in wdata["cases"]:
            case_wf[case["case_id"]] = wf
            for q in case["questions"]:
                qtype[wf][q["qid"]] = q["type"]

    out: dict[str, dict] = {wf: {} for wf in WORKFLOWS}
    for path in sorted(glob.glob(os.path.join(SRC, "*.json"))):
        raw = json.load(open(path))
        for case_id, answers in raw.items():
            wf = case_wf.get(case_id)
            if wf is None:
                print(f"  WARN: unknown case {case_id} in {os.path.basename(path)}")
                continue
            norm = {}
            for qid, val in answers.items():
                t = qtype[wf].get(qid)
                if t is None:
                    continue
                if t == "noul":
                    try:
                        p = float(val)
                    except (TypeError, ValueError):
                        continue
                    norm[qid] = {"raw": p, "kind": "noul"}
                elif t == "score":
                    norm[qid] = {"raw": val, "kind": "score"}
                else:
                    norm[qid] = {"raw": str(val), "kind": "choice"}
            out[wf][case_id] = norm

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(
        {"model": "deepseek-v4.1-flash (opencode-go, max)", "answers": out},
        open(OUT, "w"),
        indent=1,
    )

    for wf in WORKFLOWS:
        n = sum(len(a) for a in out[wf].values())
        expected = sum(len(c["reference"]) for c in full["workflows"][wf]["cases"])
        print(f"{wf:30} cases={len(out[wf])} answers={n}/{expected}")


if __name__ == "__main__":
    main()
