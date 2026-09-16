#!/usr/bin/env python3
"""Build one combined prompt file per workflow (all its cases) for API-model runs.

Output: evals/prompts_full/<workflow>.txt
The model must reply with one JSON object: {"<case_id>": {"<qid>": <answer>, ...}, ...}
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "full_eval.json")
OUT = os.path.join(HERE, "prompts_full")

HEADER = """You are a decision engine. Below are __N__ cases. Each case has an INPUT section
(documents describing the state) and a QUESTIONS section.

Answer every question for every case using ONLY the provided state. Do not refuse.
Do not use tools. Be decisive and precise.

Answer formats:
- noul   -> a number between 0 and 1: the probability that the statement is TRUE.
- score  -> an integer 0, 1, 2 or 3.
- choice -> exactly one of the listed option keys (exact string).

=== OUTPUT ===
Reply with ONLY one JSON object mapping case id -> (question id -> answer).
No prose, no markdown fences.

Example shape:
{ "case_id_1": { "some_qid": 0.7, "another_qid": 2, "choice_qid": "option_key" }, "case_id_2": { ... } }

=== CASES ===
"""

CASE = """
----- CASE {i}: {cid} -----

INPUT:
{input_text}

QUESTIONS (answer all of them):
{questions}
"""


def main() -> None:
    data = json.load(open(FULL))
    os.makedirs(OUT, exist_ok=True)
    for wf, wdata in data["workflows"].items():
        cases = wdata["cases"]
        # Very large workflows are split one case per file to keep calls reasonable
        split = wf in {"invoice_processing"}
        groups = [([c], c["case_id"]) for c in cases] if split else [(cases, wf)]

        for group, label in groups:
            chunks = [HEADER.replace("__N__", str(len(group)))]
            for i, case in enumerate(group, 1):
                q_lines = []
                for q in case["questions"]:
                    q_lines.append(f"- {q['qid']} ({q['type']}): {q['instructions']}")
                    crit = q["criteria"]
                    if isinstance(crit, dict):
                        for k, v in crit.items():
                            q_lines.append(f"    {k}: {v}")
                    elif isinstance(crit, list):
                        for idx, v in enumerate(crit):
                            q_lines.append(f"    {idx}: {v}")
                chunks.append(CASE.format(i=i, cid=case["case_id"],
                                          input_text=case["input_text"],
                                          questions="\n".join(q_lines)))
            safe = label.replace("/", "_")
            path = os.path.join(OUT, f"{safe}.txt" if split else f"{wf}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write("".join(chunks))
            size = os.path.getsize(path)
            print(f"{label:46} {len(group)} case(s), {size} chars -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
