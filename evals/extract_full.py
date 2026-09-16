#!/usr/bin/env python3
"""Extract ALL publicly shipped TypeSafe eval cases across the 4 workflows.

qid -> catalog entry mapping comes from each answer node's `questions` map
(qid -> index into the workflow's question catalog). Verified 100% coverage.

Output: evals/full_eval.json
{
  "workflows": {
    "<wf>": {
      "title": ...,
      "cases": [
        {"case_id", "name", "doc_indices", "input_text", "input_chars",
         "questions": [{"qid","type","instructions","criteria","catalog_index"}],
         "reference": {qid: {"value","probs","type"}}}
      ]
    }
  }
}
"""
from __future__ import annotations

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
T = os.path.join(HERE, "typesafe")
OUT = os.path.join(HERE, "full_eval.json")

WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]
LABELS = {
    "security_incidents": "Security Incidents",
    "agent_trace_observability": "Agent Trace Observability",
    "invoice_processing": "Invoice Processing",
    "customer_service": "Customer Service",
}


def load(wf: str) -> dict:
    raw = open(os.path.join(T, f"{wf}-cases.js"), encoding="utf-8", errors="replace").read()
    m = re.search(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", raw, re.S) or re.search(r"__VIEWER_DATA__\((.*)\)", raw, re.S)
    return json.loads(m.group(1))["eval"]


def canonical_ref_value(value, qtype: str) -> str | None:
    """Canonicalize a reference value (used when probabilities are absent)."""
    if value is None:
        return None
    if qtype == "noul":
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, (int, float)):
            return str(float(value) >= 0.5).lower()
        return str(value).lower()
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus(entry: dict, qtype: str):
    """Return (canonical_reference_value, probability_map).

    Reference format varies by workflow:
      - distribution form: sets[].probabilities present -> average + argmax
      - value form:        sets[].probabilities null    -> majority of values
    """
    sets = entry.get("sets", [])
    if not sets:
        return None, {}
    acc: dict[str, float] = {}
    values: list = []
    for s in sets:
        probs = s.get("probabilities") or {}
        for k, v in probs.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if s.get("value") is not None:
            values.append(s["value"])
    if acc:
        probs = {k: v / len(sets) for k, v in acc.items()}
        return max(probs, key=probs.get), probs
    if values:
        # majority vote over agreed values; tie -> first
        canon = [canonical_ref_value(v, qtype) for v in values]
        canon = [c for c in canon if c is not None]
        if canon:
            return max(set(canon), key=canon.count), {}
    return None, {}


def render(doc) -> str:
    return doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False)


def main() -> None:
    out = {"workflows": {}}
    total_cases = total_pairs = 0

    for wf in WORKFLOWS:
        ev = load(wf)
        docs = ev["documents"]
        catalog = ev["questions"]

        # workflow-level qid -> catalog index (union over every case's nodes;
        # the catalog is workflow-wide, so any node's map contributes)
        wf_qmap: dict[str, int] = {}
        for ex in ev["examples"]:
            case = ev["cases"][ex["case_id"]]
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    for qid, idx in (n.get("questions") or {}).items():
                        wf_qmap.setdefault(qid, int(idx))

        cases = []

        for ex in ev["examples"]:
            cid = ex["case_id"]
            case = ev["cases"][cid]

            # documents referenced by this case's nodes
            doc_idx: set[int] = set()
            for mv in case["models"].values():
                for n in mv.get("nodes", []):
                    if n.get("doc") is not None:
                        doc_idx.add(int(n["doc"]))

            # reference first (defines which qids we score)
            reference = {}
            for node_answers in case.get("reference_answers", {}).values():
                for qid, entry in node_answers.items():
                    val, probs = consensus(entry, entry["type"])
                    reference[qid] = {"value": val, "probs": probs, "type": entry["type"]}

            questions = []
            for qid, ref in reference.items():
                idx = wf_qmap.get(qid)
                if idx is None or not (0 <= idx < len(catalog)):
                    print(f"  WARN {wf}/{cid}: no catalog entry for {qid}")
                    continue
                q = catalog[idx]
                questions.append({
                    "qid": qid,
                    "type": q["type"],
                    "instructions": q["instructions"],
                    "criteria": q["criteria"],
                    "catalog_index": idx,
                })

            input_text = "\n\n".join(
                f"## Document {i}\n{render(docs[i])}" for i in sorted(doc_idx) if 0 <= i < len(docs)
            )

            cases.append({
                "case_id": cid,
                "name": ex.get("name", cid),
                "doc_indices": sorted(doc_idx),
                "input_text": input_text,
                "input_chars": len(input_text),
                "questions": questions,
                "reference": reference,
            })
            total_cases += 1
            total_pairs += len(reference)

        out["workflows"][wf] = {"title": LABELS[wf], "catalog_size": len(catalog), "cases": cases}

    json.dump(out, open(OUT, "w"))
    print(f"\nwrote {OUT}")
    for wf, data in out["workflows"].items():
        pairs = sum(len(c["reference"]) for c in data["cases"])
        chars = [c["input_chars"] for c in data["cases"]]
        missing = sum(1 for c in data["cases"] if len(c["questions"]) != len(c["reference"]))
        print(f"  {wf:30} cases={len(data['cases']):2} pairs={pairs:3} "
              f"inputs={min(chars)}-{max(chars)} chars, mapping gaps={missing}")
    print(f"TOTAL: {total_cases} cases, {total_pairs} reference pairs")


if __name__ == "__main__":
    main()
