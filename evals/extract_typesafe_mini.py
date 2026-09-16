#!/usr/bin/env python3
"""Extract a mini-eval dataset from the TypeSafe evals viewer data.

Input:  evals/typesafe/security_incidents.json  (parsed __VIEWER_DATA__ payload)
Output: evals/typesafe/mini_eval.json

Structure produced:
{
  "workflow": "security_incidents",
  "questions": [{"qid", "type", "instructions", "criteria"}],
  "cases": [
     {"case_id", "name", "docs": [{"alert", "records": [...]}],
      "model_answers": {"opus": {qid: {"value":...}}, ...},
      "reference": {qid: {"value": consensus_value, "probs": {...}}}}
  ]
}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "typesafe", "security_incidents.json")
OUT = os.path.join(HERE, "typesafe", "mini_eval.json")

# Question ids in catalog order, taken from the flowchart data-q attributes
# (verified against every case's answer keys).
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


def consensus_value(ref_entry: dict) -> tuple[str | None, dict]:
    """Mean the two reference models' probabilities; return argmax value + probs."""
    sets = ref_entry.get("sets", [])
    if not sets:
        return None, {}
    prob_sum: dict[str, float] = {}
    for s in sets:
        for k, v in s.get("probabilities", {}).items():
            prob_sum[k] = prob_sum.get(k, 0.0) + float(v)
    probs = {k: v / len(sets) for k, v in prob_sum.items()}
    if not probs:
        return None, {}
    value = max(probs, key=probs.get)
    return value, probs


def main() -> None:
    data = json.load(open(SRC))
    ev = data["eval"]

    questions = []
    for qid, q in zip(QIDS, ev["questions"]):
        questions.append({
            "qid": qid,
            "type": q["type"],
            "instructions": q["instructions"],
            "criteria": q["criteria"],
        })

    cases = []
    docs_all = ev["documents"]
    for idx, ex in enumerate(ev["examples"]):
        case_id = ex["case_id"]
        case = ev["cases"][case_id]

        # documents are stored in the same order as examples
        docs = docs_all[idx] if idx < len(docs_all) else []

        model_answers: dict[str, dict] = {}
        for mkey, mval in case["models"].items():
            answers: dict[str, dict] = {}
            for node in mval.get("nodes", []):
                for qid, ans in node.get("answers", {}).items():
                    if ans["type"] == "noul":
                        answers[qid] = {"value": bool(ans["noul"] >= 0.5), "raw": ans["noul"], "kind": "noul"}
                    elif ans["type"] == "score":
                        answers[qid] = {"value": str(ans["score"]), "raw": ans["score"], "kind": "score",
                                        "probs": ans.get("probabilities", {})}
                    elif ans["type"] == "choice":
                        answers[qid] = {"value": ans["choice"], "raw": ans["choice"], "kind": "choice",
                                        "probs": ans.get("probabilities", {})}
            model_answers[mkey] = answers

        reference: dict[str, dict] = {}
        for node_answers in case.get("reference_answers", {}).values():
            for qid, entry in node_answers.items():
                val, probs = consensus_value(entry)
                reference[qid] = {"value": val, "probs": probs, "type": entry["type"]}

        def doc_len(d):
            n = len(d.get("alert", ""))
            for r in d.get("context", {}).get("records", []):
                n += len(str(r))
            return n

        cases.append({
            "case_id": case_id,
            "name": ex.get("name", case_id),
            "label": ex.get("label", ""),
            "docs": [docs],
            "input_chars": doc_len(docs),
            "model_answers": model_answers,
            "reference": reference,
        })

    payload = {"workflow": "security_incidents", "questions": questions, "cases": cases}
    json.dump(payload, open(OUT, "w"), indent=1)

    print(f"wrote {OUT}")
    print(f"questions: {len(questions)}  cases: {len(cases)}")
    for c in cases:
        covered = len(c["reference"])
        print(f"  {c['case_id']:44} input~{c['input_chars']:6d} chars  ref_q={covered}  "
              f"models={ {m: len(a) for m, a in c['model_answers'].items()} }")


if __name__ == "__main__":
    main()
