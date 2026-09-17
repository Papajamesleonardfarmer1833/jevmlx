#!/usr/bin/env python3
"""Generate self-contained prompt files for subagent-based evaluation.

Each file contains: the alert, its context records, the question catalog entries
that the reference answers cover, and the exact JSON output contract.
Only the input state is included -- never the reference or other models' answers.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MINI = os.path.join(HERE, "typesafe", "mini_eval.json")
OUT = os.path.join(HERE, "prompts")

HEADER = """You are a Security Operations decision engine.
You will be given an alert and the context records joined to it, then a catalog of questions.
Answer every question using ONLY the provided state. Be precise. Do not refuse.
Do not use any tools.

=== INPUT ===

Alert:
{alert}

Context records:
{records}

=== QUESTIONS ===
{questions}

=== OUTPUT CONTRACT ===

Reply with ONLY a single JSON object, one key per question id. No prose, no markdown fences.

- noul   -> a number between 0 and 1: the probability that the statement is TRUE.
- score  -> an integer 0, 1, 2 or 3.
- choice -> exactly one of the listed option keys.

Expected keys and types:
{contract}
"""


def main() -> None:
    mini = json.load(open(MINI))
    os.makedirs(OUT, exist_ok=True)

    for i, case in enumerate(mini["cases"], 1):
        doc = case["docs"][0]
        relevant = [q for q in mini["questions"] if q["qid"] in case["reference"]]

        q_lines = []
        contract = {}
        for q in relevant:
            q_lines.append(f"- {q['qid']} ({q['type']}): {q['instructions']}")
            crit = q["criteria"]
            if isinstance(crit, dict):
                for k, v in crit.items():
                    q_lines.append(f"    {k}: {v}")
                if q["type"] == "choice":
                    contract[q["qid"]] = f"choice, one of {list(crit.keys())}"
                else:
                    contract[q["qid"]] = "noul, number 0..1"
            else:
                for idx, v in enumerate(crit):
                    q_lines.append(f"    {idx}: {v}")
                contract[q["qid"]] = "score, integer 0..3"

        text = HEADER.format(
            alert=doc.get("alert", "").strip(),
            records="\n".join(str(r) for r in doc.get("context", {}).get("records", [])),
            questions="\n".join(q_lines),
            contract="\n".join(f'  "{k}": {v}' for k, v in contract.items()),
        )

        fname = f"case_{i:02d}_{case['case_id']}.txt"
        with open(os.path.join(OUT, fname), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{fname}: {len(text)} chars, {len(relevant)} questions")

    print(f"\nwrote {len(mini['cases'])} prompt files to {OUT}")


if __name__ == "__main__":
    main()
