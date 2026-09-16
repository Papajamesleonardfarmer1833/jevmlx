#!/usr/bin/env python3
"""Full scoring: all 4 workflows, 20 cases, 373 reference pairs.

Models: opus / sol / typesafe(Jev) [published], deepseek-v4.1-flash [subagents],
        local-qwen3-8b [parallel-decisions engine].
Metric: agreement with the consensus argmax (mean of the two reference models).

Output: evals/results/full-summary.json + printed tables.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "full_eval.json")
PUB = os.path.join(HERE, "published_answers.json")
RESULTS = os.path.join(HERE, "results")
EXTRA = {
    "deepseek-v4.1-flash": os.path.join(RESULTS, "full-deepseek-v4.1-flash.json"),
    "local-qwen3-8b": os.path.join(RESULTS, "full-local-qwen3-8b.json"),
    "local-qwen2.5-7b": os.path.join(RESULTS, "full-local-qwen2.5-7b.json"),
}
WORKFLOWS = ["security_incidents", "agent_trace_observability", "invoice_processing", "customer_service"]
LABEL = {"security_incidents": "Security", "agent_trace_observability": "AgentTrace",
         "invoice_processing": "Invoice", "customer_service": "CustomerSvc"}


def canonical(value, kind: str, qtype: str) -> str | None:
    if value is None:
        return None
    if qtype == "noul":
        try:
            p = float(value)
        except (TypeError, ValueError):
            return None
        return str(p >= 0.5).lower()
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def consensus(ref: dict) -> str | None:
    """Reference answer: argmax of the consensus distribution, else the agreed value."""
    probs = ref.get("probs") or {}
    if probs:
        return str(max(probs, key=probs.get))
    value = ref.get("value")
    return str(value) if value is not None else None


def qid_types(full: dict) -> dict[str, dict[str, str]]:
    """workflow -> {qid: type} built from every case's mapped questions."""
    out: dict[str, dict[str, str]] = {}
    for wf, wdata in full["workflows"].items():
        out[wf] = {}
        for case in wdata["cases"]:
            for q in case["questions"]:
                out[wf][q["qid"]] = q["type"]
    return out


def load_extra(path: str, qtype_by_wf) -> dict:
    data = json.load(open(path))["answers"]
    out = {}
    for wf, cases in data.items():
        out[wf] = {}
        for cid, answers in cases.items():
            out[wf][cid] = {}
            for qid, a in answers.items():
                t = qtype_by_wf[wf].get(qid)
                if t is None:
                    continue
                c = canonical(a.get("raw"), a.get("kind", ""), t)
                if c is not None:
                    out[wf][cid][qid] = c
    return out


def main() -> None:
    full = json.load(open(FULL))
    published = json.load(open(PUB))

    qtype_by_wf: dict[str, dict[str, str]] = {}
    for wf, wdata in full["workflows"].items():
        qtype_by_wf[wf] = {}
        for case in wdata["cases"]:
            for q in case["questions"]:
                qtype_by_wf[wf][q["qid"]] = q["type"]

    # published models -> canonical
    models: dict[str, dict] = {}
    for mkey in ("opus", "sol", "typesafe"):
        m: dict = {}
        for wf, cases in published.items():
            m[wf] = {}
            for cid, per_model in cases.items():
                answers = per_model.get(mkey, {})
                m[wf][cid] = {}
                for qid, a in answers.items():
                    t = qtype_by_wf[wf].get(qid)
                    if t is None:
                        continue
                    c = canonical(a.get("raw"), a.get("kind", ""), t)
                    if c is not None:
                        m[wf][cid][qid] = c
        models[mkey] = m

    for name, path in EXTRA.items():
        if os.path.isfile(path):
            models[name] = load_extra(path, qtype_by_wf)
        else:
            print(f"(skipping {name}: {path} not found)")

    # ---- scoring ----
    # (a) per workflow, over answered pairs
    rows = []
    for name, m in models.items():
        wf_stats = {}
        for wf, wdata in full["workflows"].items():
            ok = n = 0
            for case in wdata["cases"]:
                cid = case["case_id"]
                for qid, ref in case["reference"].items():
                    want = consensus(ref)
                    got = m.get(wf, {}).get(cid, {}).get(qid)
                    if want is None or got is None:
                        continue
                    ok += got == want
                    n += 1
            wf_stats[wf] = (ok, n)
        tot_ok, tot_n = sum(x for x, y in wf_stats.values()), sum(y for x, y in wf_stats.values())
        rows.append((name, wf_stats, tot_ok, tot_n))

    # (b) strict common subset overall + per workflow
    common: dict[str, list] = {}
    for wf, wdata in full["workflows"].items():
        pairs = []
        for case in wdata["cases"]:
            cid = case["case_id"]
            for qid, ref in case["reference"].items():
                if consensus(ref) is None:
                    continue
                if all(qid in models[m].get(wf, {}).get(cid, {}) for m in models):
                    pairs.append((cid, qid))
        common[wf] = pairs

    print()
    print("=== Agreement with TypeSafe's reference (consensus of GPT-6 Astra + Fable 5.1) ===")
    hdr = f"{'model':<22} {'overall':>13} " + " ".join(f"{LABEL[w]:>15}" for w in WORKFLOWS)
    print(hdr)
    print("-" * len(hdr))
    for name, wf_stats, tot_ok, tot_n in sorted(rows, key=lambda r: -(r[2] / r[3] if r[3] else 0)):
        cells = []
        for wf in WORKFLOWS:
            ok, n = wf_stats[wf]
            cells.append(f"{ok}/{n}" if n else "-")
        acc = f"{tot_ok / tot_n * 100:.1f}%" if tot_n else "n/a"
        print(f"{name:<22} {acc:>6} {tot_ok:>3}/{tot_n:<3} " + " ".join(f"{c:>15}" for c in cells))

    print()
    print("=== Strict common subset (pairs answered by EVERY model) ===")
    print(f"{'model':<22} {'overall':>12} " + " ".join(f"{LABEL[w]:>12}" for w in WORKFLOWS))
    print("-" * 92)
    for name in models:
        cells = []
        for wf in WORKFLOWS:
            ok = n = 0
            for cid, qid in common[wf]:
                case = next(c for c in full["workflows"][wf]["cases"] if c["case_id"] == cid)
                want = consensus(case["reference"][qid])
                got = models[name].get(wf, {}).get(cid, {}).get(qid)
                if want is not None and got is not None:
                    ok += got == want
                    n += 1
            cells.append(f"{ok}/{n}" if n else "-")
        # overall common
        ok_all = n_all = 0
        for wf in WORKFLOWS:
            for cid, qid in common[wf]:
                case = next(c for c in full["workflows"][wf]["cases"] if c["case_id"] == cid)
                want = consensus(case["reference"][qid])
                got = models[name].get(wf, {}).get(cid, {}).get(qid)
                if want is not None and got is not None:
                    ok_all += got == want
                    n_all += 1
        acc = f"{ok_all / n_all * 100:.1f}%" if n_all else "n/a"
        print(f"{name:<22} {acc:>12} " + " ".join(f"{c:>12}" for c in cells))

    print()
    print("Common-subset sizes:", {LABEL[w]: len(p) for w, p in common.items()})

    # per-model strict common-subset stats
    common_rows = {}
    for name in models:
        per_wf = {}
        ok_all = n_all = 0
        for wf in WORKFLOWS:
            ok = n = 0
            for cid, qid in common[wf]:
                case = next(c for c in full["workflows"][wf]["cases"] if c["case_id"] == cid)
                want = consensus(case["reference"][qid])
                got = models[name].get(wf, {}).get(cid, {}).get(qid)
                if want is not None and got is not None:
                    ok += got == want
                    n += 1
            per_wf[wf] = [ok, n]
            ok_all += ok
            n_all += n
        common_rows[name] = {"overall": [ok_all, n_all], "per_workflow": per_wf}

    out = {
        "rows": [{"model": n, "overall": [ok, tot], "per_workflow": {w: list(v) for w, v in s.items()}}
                 for n, s, ok, tot in rows],
        "common_subset": common_rows,
        "common_subset_size": {w: len(p) for w, p in common.items()},
    }
    json.dump(out, open(os.path.join(RESULTS, "full-summary.json"), "w"), indent=1)
    print(f"\nsaved {os.path.join(RESULTS, 'full-summary.json')}")


if __name__ == "__main__":
    main()
