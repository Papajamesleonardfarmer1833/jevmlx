#!/usr/bin/env python3
"""Record subagent answers for the TypeSafe mini-eval into the standard result format.

Usage:
  python record_subagent_answers.py            # reads answers.json (hand-maintained),
#                                              # writes results/deepseek-v4.1-flash.json
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "subagent_answers.json")
OUT = os.path.join(HERE, "results", "deepseek-v4.1-flash.json")

# Question type per qid, from the catalog
QIDS = [
    "is_true_positive",
    "context_explains_activity",
    "evidence_strength",
    "credentials_exposed",
    "session_in_attacker_hands",
    "malicious_content_in_mailboxes",
    "attacker_persistence_present",
    "malicious_process_running",
    "outbound_channel_active",
    "attacker_modified_configuration",
    "activity_ongoing",
    "spread_beyond_initial_entity",
    "affected_scope",
    "attack_type",
]
NOUL = {
    "is_true_positive",
    "context_explains_activity",
    "credentials_exposed",
    "session_in_attacker_hands",
    "malicious_content_in_mailboxes",
    "attacker_persistence_present",
    "malicious_process_running",
    "outbound_channel_active",
    "attacker_modified_configuration",
    "activity_ongoing",
    "spread_beyond_initial_entity",
}
SCORE = {"evidence_strength"}
CHOICE = {"affected_scope", "attack_type"}


def normalize(qid: str, raw) -> dict:
    if qid in NOUL:
        p = float(raw)
        return {"value": p >= 0.5, "raw": p, "kind": "noul"}
    if qid in SCORE:
        lvl = max(0, min(3, int(round(float(raw)))))
        return {"value": str(lvl), "raw": lvl, "kind": "score"}
    return {"value": str(raw), "raw": raw, "kind": "choice"}


def main() -> None:
    data = json.load(open(RAW))
    answers = {}
    for case_id, raw_answers in data.items():
        answers[case_id] = {qid: normalize(qid, v) for qid, v in raw_answers.items()}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(
        {"model": "deepseek-v4.1-flash (opencode-go, max)", "answers": answers},
        open(OUT, "w"),
        indent=1,
    )
    print(f"wrote {OUT}: {len(answers)} cases, {sum(len(a) for a in answers.values())} answers")


if __name__ == "__main__":
    main()
