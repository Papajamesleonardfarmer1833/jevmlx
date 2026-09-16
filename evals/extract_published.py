#!/usr/bin/env python3
"""Collect the published model answers (opus / sol / typesafe-Jev) for all cases.

Output: evals/published_answers.json
  {"<wf>": {"<case_id>": {"opus": {qid: raw}, "sol": {...}, "typesafe": {...}}}}
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "typesafe")
OUT = os.path.join(HERE, "published_answers.json")
WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]


def load(wf: str) -> dict:
    raw = open(os.path.join(T, f"{wf}-cases.js"), encoding="utf-8", errors="replace").read()
    m = re.search(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", raw, re.S) or re.search(r"__VIEWER_DATA__\((.*)\)", raw, re.S)
    return json.loads(m.group(1))["eval"]


def main() -> None:
    out: dict = {}
    for wf in WORKFLOWS:
        ev = load(wf)
        wf_out: dict = {}
        for ex in ev["examples"]:
            cid = ex["case_id"]
            case = ev["cases"][cid]
            per_model = {}
            for mkey, mv in case["models"].items():
                answers = {}
                for n in mv.get("nodes", []):
                    for qid, ans in (n.get("answers") or {}).items():
                        if ans["type"] == "noul":
                            answers[qid] = {"raw": ans.get("noul"), "kind": "noul"}
                        elif ans["type"] == "score":
                            answers[qid] = {"raw": ans.get("score"), "kind": "score"}
                        elif ans["type"] == "choice":
                            answers[qid] = {"raw": ans.get("choice"), "kind": "choice"}
                per_model[mkey] = answers
            wf_out[cid] = per_model
        out[wf] = wf_out

    json.dump(out, open(OUT, "w"), indent=1)
    for wf, cases in out.items():
        n = sum(len(a) for c in cases.values() for a in c.values())
        print(f"{wf:30} cases={len(cases)} answers={n}")


if __name__ == "__main__":
    main()
