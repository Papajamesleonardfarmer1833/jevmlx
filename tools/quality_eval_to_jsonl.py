#!/usr/bin/env python3
"""Convert quality-eval/cases.json (labeled) to the calibrator's JSONL format.

Usage: .venv/bin/python tools/quality_eval_to_jsonl.py > quality-eval/cases.jsonl
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "..", "quality-eval", "cases.json")

with open(CASES, encoding="utf-8") as f:
    data = json.load(f)

for family in data["families"].values():
    for case in family["cases"]:
        print(json.dumps({"schema": family["schema"],
                          "context": case["context"],
                          "labels": case["labels"]}))
