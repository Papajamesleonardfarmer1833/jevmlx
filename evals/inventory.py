#!/usr/bin/env python3
"""Inventory the full public eval data across all 4 TypeSafe workflows.

Answers: how many cases, which questions have reference answers, input sizes,
which models answered what. Outputs evals/typesafe/inventory.json + a printed table.
"""

from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "typesafe")

WORKFLOWS = [
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
]


def load(wf: str) -> dict:
    raw = open(os.path.join(T, f"{wf}-cases.js"), encoding="utf-8", errors="replace").read()
    m = re.search(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", raw, re.S) or re.search(
        r"__VIEWER_DATA__\((.*)\)", raw, re.S
    )
    return json.loads(m.group(1))["eval"]


def doc_chars(doc) -> int:
    return len(json.dumps(doc))


def main() -> None:
    inv = {}
    grand_pairs = 0
    for wf in WORKFLOWS:
        ev = load(wf)
        per_case = []
        for i, ex in enumerate(ev["examples"]):
            cid = ex["case_id"]
            case = ev["cases"][cid]
            ref = case.get("reference_answers", {})
            n_ref = sum(len(v) for v in ref.values())
            models = {
                m: sum(len(n.get("answers", {})) for n in v.get("nodes", []))
                for m, v in case["models"].items()
            }
            docs = ev["documents"]
            # documents may be 1:1 with cases or multiple per case; try index if same count
            if len(docs) == len(ev["examples"]):
                doc = docs[i]
                chars = doc_chars(doc)
            else:
                chars = None
            per_case.append(
                {
                    "case_id": cid,
                    "name": ex.get("name"),
                    "input_chars": chars,
                    "reference_pairs": n_ref,
                    "model_answers": models,
                    "reference_nodes": {k: len(v) for k, v in ref.items()},
                }
            )
            grand_pairs += n_ref
        inv[wf] = {
            "n_cases_total": ev["n_cases"],
            "shipped_examples": len(ev["examples"]),
            "catalog_questions": len(ev["questions"]),
            "cases": per_case,
        }

    json.dump(inv, open(os.path.join(T, "inventory.json"), "w"), indent=1)

    print(
        f"{'workflow':<30} {'case':<40} {'in_chars':>9} {'ref_pairs':>9}  models(opus/sol/jev)"
        f"ref_nodes"
    )
    print("-" * 140)
    total = 0
    for wf, data in inv.items():
        for c in data["cases"]:
            m = c["model_answers"]
            print(
                f"{wf:<30} {c['case_id'][:38]:<40} {str(c['input_chars']):>9}"
                f"{c['reference_pairs']:>9}"
                f""
                f"{m.get('opus')}/{m.get('sol')}/{m.get('typesafe')}   {c['reference_nodes']}"
            )
            total += c["reference_pairs"]
        print(
            f"  (catalog questions: {data['catalog_questions']}, total cases in full eval:"
            f"{data['n_cases_total']})"
        )
    print(f"\nTOTAL reference question-pairs across shipped cases: {total}")
    print(f"Total shipped cases: {sum(d['shipped_examples'] for d in inv.values())}")


if __name__ == "__main__":
    main()
