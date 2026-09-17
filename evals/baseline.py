#!/usr/bin/env python3
"""Baseline analysis: majority-class agreement with the reference.

For each workflow, the majority baseline answers the most common reference value per
question type (or per question id where there is only one question id). This gives the
floor that a trivial model would reach, so the real numbers can be read against it.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "full_eval.json")

WORKFLOWS = [
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
]


def main() -> None:
    full = json.load(open(FULL))
    print(f"{'workflow':<30} {'pairs':>6} {'per-qid majority':>18} {'per-type majority':>19}")
    print("-" * 80)
    for wf in WORKFLOWS:
        wdata = full["workflows"][wf]

        # how often does each qid appear (across cases)?
        appear = Counter()
        for case in wdata["cases"]:
            for qid in case["reference"]:
                appear[qid] += 1

        by_qid = defaultdict(Counter)
        by_type = defaultdict(Counter)
        scoreable = []  # (case, qid) pairs where a per-qid majority exists (qid seen >=2x)
        for case in wdata["cases"]:
            for qid, ref in case["reference"].items():
                v = ref.get("value")
                if v is None:
                    continue
                t = ref["type"]
                by_qid[qid][v] += 1
                by_type[t][v] += 1
                if appear[qid] >= 2:
                    scoreable.append((qid, v, t))

        per_qid_ok = sum(1 for qid, v, t in scoreable if v == by_qid[qid].most_common(1)[0][0])
        per_type_ok = sum(1 for qid, v, t in scoreable if v == by_type[t].most_common(1)[0][0])
        n = len(scoreable)
        print(f"{wf:<30} {n:>6} {per_qid_ok / n * 100:>16.1f}% {per_type_ok / n * 100:>18.1f}%")

    # value distribution per workflow to spot degenerate answer sets
    print()
    print("Reference value distribution per workflow (top 6):")
    for wf in WORKFLOWS:
        c = Counter()
        for case in full["workflows"][wf]["cases"]:
            for ref in case["reference"].values():
                if ref.get("value") is not None:
                    c[str(ref["value"])] += 1
        top = ", ".join(f"{k}={v}" for k, v in c.most_common(6))
        print(f"  {wf:<30} {top}")


if __name__ == "__main__":
    main()
