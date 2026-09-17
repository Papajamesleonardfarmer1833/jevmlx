#!/usr/bin/env python3
"""Convert the repo's labeled cases (benchmarks/cases.json) to the calibrator's JSONL format.

Usage: .venv/bin/python benchmarks/to_jsonl.py > cases.jsonl
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases.json")

with open(CASES, encoding="utf-8") as f:
    data = json.load(f)

for family in data["families"].values():
    for case in family["cases"]:
        print(
            json.dumps(
                {"schema": family["schema"], "context": case["context"], "labels": case["labels"]}
            )
        )
