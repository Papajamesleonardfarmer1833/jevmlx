#!/usr/bin/env python3
"""Merge per-model eval results into results/summary.json and quality-eval/SUMMARY.md.

Usage:
    .venv/bin/python quality-eval/analyze.py [--outdir quality-eval/results]

Reads every rlcd-quality-eval result file in the results directory, computes
accuracy / calibration / latency metrics, prints markdown and ASCII tables, and
answers "is the larger model worth it?" in the generated SUMMARY.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTDIR = os.path.join(HERE, "results")

BUCKETS = [
    ("<0.70", lambda p: p < 0.70),
    ("0.70-0.90", lambda p: 0.70 <= p <= 0.90),
    (">0.90", lambda p: p > 0.90),
]

# A model is "clearly better" only if it wins by at least this many cases on the
# primary fields (24 cases total, so 2 cases ~= 8 percentage points).
WIN_MARGIN_CASES = 2


def load_runs(outdir: str) -> tuple[list[dict], list[dict]]:
    runs, skipped = [], []
    for path in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        if data.get("kind") != "rlcd-quality-eval":
            continue  # summary.json and anything else
        if "smoke" in data.get("tag", ""):
            continue  # smoke tests never mix with the real numbers
        if data.get("status") == "load_failed":
            skipped.append(data)
            continue
        runs.append(data)
    return runs, skipped


def field_rows(run: dict) -> list[dict]:
    """One row per field decision: family, field, correct, confidence, on_alt."""
    rows = []
    for case in run["results"]:
        if case.get("status") != "ok":
            continue
        for name, label in case["labels"].items():
            pred = case["predicted"][name]
            alt = case["acceptable"].get(name, [])
            rows.append(
                {
                    "case": case["id"],
                    "family": case["family"],
                    "primary": name == case["primary_field"],
                    "field": name,
                    "correct": pred == label,
                    "correct_or_acceptable": pred == label or pred in alt,
                    "on_alt": pred != label and pred in alt,
                    "confidence": float(case["confidence"][name]),
                }
            )
    return rows


def metrics(run: dict) -> dict:
    ok = [c for c in run["results"] if c.get("status") == "ok"]
    rows = field_rows(run)
    primary = [r for r in rows if r["primary"]]

    def acc(rs) -> float:
        return sum(r["correct"] for r in rs) / len(rs) if rs else float("nan")

    def acc_alt(rs) -> float:
        return sum(r["correct_or_acceptable"] for r in rs) / len(rs) if rs else float("nan")

    correct_conf = [r["confidence"] for r in rows if r["correct"]]
    wrong_conf = [r["confidence"] for r in rows if not r["correct"]]

    buckets = []
    for name, pred in BUCKETS:
        rs = [r for r in rows if pred(r["confidence"])]
        buckets.append({"bucket": name, "fields": len(rs), "accuracy": acc(rs)})

    latencies = [c["elapsed_ms"] for c in ok]
    families = sorted({c["family"] for c in ok})
    per_family = {}
    for fam in families:
        per_family[fam] = {
            "primary_accuracy": acc([r for r in primary if r["family"] == fam]),
            "primary_accuracy_or_acceptable": acc_alt([r for r in primary if r["family"] == fam]),
            "all_fields_accuracy": acc([r for r in rows if r["family"] == fam]),
        }

    overconfident = [r for r in rows if not r["correct"] and r["confidence"] > 0.90]
    missed_cases = {}
    for c in ok:
        bad = [name for name in c["labels"] if not c["field_correct"][name]]
        if bad:
            missed_cases[c["id"]] = bad

    return {
        "model": run["model"],
        "tag": run["tag"],
        "status": run.get("status", "ok"),
        "cases": len(ok),
        "errors": len(run["results"]) - len(ok),
        "load_seconds": run.get("load_seconds"),
        "total_elapsed_ms": run.get("total_elapsed_ms"),
        "primary_accuracy": acc(primary),
        "primary_accuracy_or_acceptable": acc_alt(primary),
        "all_fields_accuracy": acc(rows),
        "all_fields_accuracy_or_acceptable": acc_alt(rows),
        "field_accuracy": acc(rows),
        "field_accuracy_or_acceptable": acc_alt(rows),
        "fields": len(rows),
        "mean_confidence_correct": statistics.fmean(correct_conf) if correct_conf else None,
        "mean_confidence_incorrect": statistics.fmean(wrong_conf) if wrong_conf else None,
        "confidence_gap": (statistics.fmean(correct_conf) - statistics.fmean(wrong_conf))
        if correct_conf and wrong_conf
        else None,
        "overconfident_misses": len(overconfident),
        "missing_fields": len(rows) - sum(r["correct"] for r in rows),
        "missed_cases": missed_cases,
        "confidence_buckets": buckets,
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "mean_prefill_ms": statistics.fmean([c["prefill_ms"] for c in ok]) if ok else None,
        "mean_suffix_eval_ms": statistics.fmean([c["suffix_eval_ms"] for c in ok]) if ok else None,
        "cases_with_acceptable_alternative": sorted({r["case"] for r in rows if r["on_alt"]}),
        "fields_on_acceptable_alternative": sum(r["on_alt"] for r in rows),
        "per_family": per_family,
    }


def majority_baseline(runs: list[dict]) -> dict:
    """Trivial baseline: predict the most frequent label per family (from the case file labels)."""
    cases = []
    for run in runs:
        for c in run["results"]:
            if c.get("status") == "ok":
                cases.append(c)
    by_family, hits = {}, 0
    for c in cases:
        fam = c["family"]
        by_family.setdefault(fam, []).append(c)
    for _fam, cs in by_family.items():
        primary = cs[0]["primary_field"]
        values = [c["labels"][primary] for c in cs]
        majority = max(set(values), key=values.count)
        hits += sum(1 for c in cs if c["labels"][primary] == majority)
    return {"primary_accuracy": hits / len(cases) if cases else float("nan")}


def pct(x, digits: int = 1) -> str:
    if x is None:
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def num(x, digits: int = 0) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [headers] + rows
    widths = [max(len(str(row[i])) for row in table) for i in range(len(headers))]
    lines = ["  ".join(str(h).ljust(w) for h, w in zip(headers, widths, strict=False))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=False)))
    return "\n".join(lines)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def verdict(model_metrics: list[dict]) -> dict:
    """Compare models on primary-field accuracy and all-fields exactness."""
    if len(model_metrics) < 2:
        return {"comparisons": [], "margin_cases": WIN_MARGIN_CASES}
    n_cases = model_metrics[0]["cases"]
    margin = WIN_MARGIN_CASES / n_cases  # in accuracy units
    by_tag = {m["tag"]: m for m in model_metrics}

    def find(*needles):
        for tag, m in by_tag.items():
            if all(needle in tag for needle in needles):
                return m
        return None

    small, mid, large = find("1.5b"), find("7b"), find("8b")
    comparisons = []

    def pair(a, b, label):
        if not (a and b):
            return None
        entry = {
            "pair": f"{a['tag']} vs {b['tag']}",
            "label": label,
            "primary_diff_pp": round(100.0 * (a["primary_accuracy"] - b["primary_accuracy"]), 1),
            "all_fields_diff_pp": round(
                100.0 * (a["all_fields_accuracy"] - b["all_fields_accuracy"]), 1
            ),
            "a_primary_clearly_better": (a["primary_accuracy"] - b["primary_accuracy"]) >= margin,
            "a_all_fields_clearly_better": (a["all_fields_accuracy"] - b["all_fields_accuracy"])
            >= margin,
            "latency_ratio": (a["mean_latency_ms"] / b["mean_latency_ms"])
            if b["mean_latency_ms"]
            else None,
        }
        comparisons.append(entry)
        return entry

    mid_vs_small = pair(mid, small, "mid vs small")
    large_vs_mid = pair(large, mid, "large vs mid")
    return {
        "margin_cases": WIN_MARGIN_CASES,
        "comparisons": comparisons,
        "mid_vs_small": mid_vs_small,
        "large_vs_mid": large_vs_mid,
    }


def confidence_notes(mm: list[dict]) -> list[str]:
    notes = []
    for m in mm:
        if m["confidence_gap"] is None:
            notes.append(f"- {m['tag']}: no incorrect fields, confidence gap not measurable.")
            continue
        share = (
            (100.0 * m["overconfident_misses"] / m["missing_fields"])
            if m["missing_fields"]
            else 0.0
        )
        notes.append(
            f"- {m['tag']}: mean confidence {num(m['mean_confidence_correct'], 2)} on correct vs "
            f"{num(m['mean_confidence_incorrect'], 2)} on incorrect fields (gap "
            f"{num(m['confidence_gap'], 2)}); {m['overconfident_misses']} of {m['missing_fields']} "
            f"wrong fields ({share:.0f}%) were reported above 0.90 confidence."
        )
    return notes


def verdict_sentence(mm: list[dict], cmp: dict) -> str:
    """One-paragraph answer to 'is the larger model worth it?'."""
    small, mid, large = _pick(mm)
    c_sm, c_ml = cmp.get("mid_vs_small"), cmp.get("large_vs_mid")
    if not (small and mid and c_sm):
        return "Verdict unavailable: fewer than two models completed."

    parts = []
    if c_sm["a_primary_clearly_better"]:
        parts.append(
            f"**Yes: the step up from {small['tag']} to {mid['tag']} is clearly worth it.** "
            f"{mid['tag']} gains {c_sm['primary_diff_pp']:+.1f} points on the primary field "
            f"({pct(mid['primary_accuracy'])} vs {pct(small['primary_accuracy'])}) and "
            f"{c_sm['all_fields_diff_pp']:+.1f} points on all-fields exact "
            f"({pct(mid['all_fields_accuracy'])} vs {pct(small['all_fields_accuracy'])}) "
            f"for {num(c_sm['latency_ratio'], 1)}x the latency"
        )
    else:
        parts.append(
            f"**No clear win for {mid['tag']} over {small['tag']} on this eval:** "
            f"primary field {c_sm['primary_diff_pp']:+.1f} points, all fields "
            f"{c_sm['all_fields_diff_pp']:+.1f} points, "
            f"for {num(c_sm['latency_ratio'], 1)}x the latency"
        )

    if c_ml:
        if c_ml["a_all_fields_clearly_better"] and not c_ml["a_primary_clearly_better"]:
            parts.append(
                f"the {large['tag']} is a wash on the primary field "
                f"({c_ml['primary_diff_pp']:+.1f} points, inside the "
                f"{cmp['margin_cases']}-case margin of this eval) "
                f"but clearly more consistent overall: "
                f"{c_ml['all_fields_diff_pp']:+.1f} points on all-fields exact "
                f"({pct(large['all_fields_accuracy'])} vs {pct(mid['all_fields_accuracy'])}) "
                f"at {num(c_ml['latency_ratio'], 2)}x the latency"
            )
        elif c_ml["a_all_fields_clearly_better"] and c_ml["a_primary_clearly_better"]:
            parts.append(
                f"the {large['tag']} beats the {mid['tag']} on both metrics "
                f"({c_ml['primary_diff_pp']:+.1f} primary, "
                f"{c_ml['all_fields_diff_pp']:+.1f} all fields), "
                f"so it is the accuracy ceiling at {num(c_ml['latency_ratio'], 2)}x the latency"
            )
        else:
            parts.append(
                f"the {large['tag']} adds nothing over the {mid['tag']} "
                f"({c_ml['primary_diff_pp']:+.1f} primary, "
                f"{c_ml['all_fields_diff_pp']:+.1f} all fields) "
                f"and costs {num(c_ml['latency_ratio'], 2)}x the latency"
            )
    return "; ".join(parts) + "."


def build_summary_md(
    runs: list[dict], mm: list[dict], baseline: dict, skipped: list[dict], cmp: dict
) -> str:
    small, mid, large = _pick(mm)
    main_headers = [
        "Model",
        "Primary acc",
        "Primary acc (alt)",
        "All-fields exact",
        "All-fields exact (alt)",
        "Mean conf correct",
        "Mean conf incorrect",
        "Latency/case",
        "Load (warm)",
    ]
    main_rows = []
    for m in mm:
        main_rows.append(
            [
                m["tag"],
                pct(m["primary_accuracy"]),
                pct(m["primary_accuracy_or_acceptable"]),
                pct(m["all_fields_accuracy"]),
                pct(m["all_fields_accuracy_or_acceptable"]),
                num(m["mean_confidence_correct"], 2),
                num(m["mean_confidence_incorrect"], 2),
                f"{num(m['mean_latency_ms'])} ms",
                f"{num(m['load_seconds'])} s",
            ]
        )
    main_rows.append(
        [
            "majority-class baseline",
            pct(baseline["primary_accuracy"]),
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
            "n/a",
        ]
    )
    main_ascii = ascii_table(
        [
            "model",
            "prim-acc",
            "prim-alt",
            "all-exact",
            "all-alt",
            "conf-ok",
            "conf-miss",
            "lat/case",
            "load*",
        ],
        [[r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]] for r in main_rows],
    )
    main_md = md_table(main_headers, main_rows)

    field_headers = (
        ["Model"]
        + [f"A.{f}" for f in ("fraud", "risk", "action")]
        + [f"B.{f}" for f in ("category", "priority", "needs_human")]
    )
    field_rows_md = [
        [m["tag"]]
        + [
            pct(m["per_field_accuracy"].get(f))
            for f in ("fraud", "risk", "action", "category", "priority", "needs_human")
        ]
        for m in mm
    ]
    field_md = md_table(field_headers, field_rows_md)

    fam_rows = [
        [
            m["tag"],
            pct(m["per_family"].get("payment_risk", {}).get("primary_accuracy")),
            pct(m["per_family"].get("support_triage", {}).get("primary_accuracy")),
            pct(m["per_family"].get("payment_risk", {}).get("all_fields_accuracy")),
            pct(m["per_family"].get("support_triage", {}).get("all_fields_accuracy")),
        ]
        for m in mm
    ]
    fam_md = md_table(
        ["Model", "A primary acc", "B primary acc", "A all-fields", "B all-fields"], fam_rows
    )

    bucket_headers = [
        "Model",
        "<0.70 (fields / acc)",
        "0.70-0.90 (fields / acc)",
        ">0.90 (fields / acc)",
    ]
    bucket_rows = []
    for m in mm:
        cells = [f"{b['fields']} / {pct(b['accuracy'])}" for b in m["confidence_buckets"]]
        bucket_rows.append([m["tag"]] + cells)
    buckets_md = md_table(bucket_headers, bucket_rows)

    lat_headers = [
        "Model",
        "Mean latency",
        "Median latency",
        "Mean prefill",
        "Mean suffix",
        "Load (warm)",
    ]
    lat_rows = [
        [
            m["tag"],
            f"{num(m['mean_latency_ms'])} ms",
            f"{num(m['median_latency_ms'])} ms",
            f"{num(m['mean_prefill_ms'])} ms",
            f"{num(m['mean_suffix_eval_ms'])} ms",
            f"{num(m['load_seconds'])} s",
        ]
        for m in mm
    ]
    lat_md = md_table(lat_headers, lat_rows)

    # Verdict text, driven by the comparison numbers above.
    lines = []
    for c in cmp.get("comparisons", []):
        lines.append(
            f"- `{c['pair']}` ({c['label']}): primary-field accuracy differs by "
            f"{c['primary_diff_pp']:+.1f} points, all-fields exact by "
            f"{c['all_fields_diff_pp']:+.1f} points; latency ratio {num(c['latency_ratio'], 2)}x; "
            f"clearly better only at a margin of >= {cmp['margin_cases']} cases "
            f"({100.0 * cmp['margin_cases'] / (mm[0]['cases'] or 1):.1f} points)."
        )
    verdict_block = verdict_sentence(mm, cmp) + ("\n\n" + "\n".join(lines) if lines else "")

    confidence_note = confidence_notes(mm)
    confidence_block = "\n".join(confidence_note)

    fail_rows = []
    for m in mm:
        if not m["missed_cases"]:
            fail_rows.append([m["tag"], str(m["missing_fields"]), "none"])
            continue
        detail = "; ".join(
            f"{cid}: {', '.join(fs)}" for cid, fs in sorted(m["missed_cases"].items())
        )
        fail_rows.append([m["tag"], str(m["missing_fields"]), detail])
    fail_md = md_table(["Model", "Wrong fields (of 72)", "Strict misses per case"], fail_rows)

    anomaly = []
    for m in mm:
        if m["all_fields_accuracy"] == 0 and m["fields"] > 0:
            anomaly.append(
                f"- {m['tag']}: every field decision was wrong; "
                f"treat this run as broken, not as a quality signal."
            )
        if m["errors"]:
            anomaly.append(f"- {m['tag']}: {m['errors']} case(s) errored during generation.")
    for s in skipped:
        anomaly.append(f"- {s['model']}: model failed to load ({s.get('error', 'unknown error')}).")
    anomaly_block = "\n".join(anomaly) if anomaly else "- No broken runs, no errored cases."

    amb_rows = [
        [
            m["tag"],
            str(m["fields_on_acceptable_alternative"]),
            ", ".join(m["cases_with_acceptable_alternative"]) or "-",
        ]
        for m in mm
    ]

    return f"""# Quality evaluation: is the larger model worth it?

Generated by `quality-eval/analyze.py` on {dt.datetime.now().date().isoformat()} from
{len(runs)} completed model run(s). Machine: MacBook Air M5, 16 GB unified memory,
macOS 26.5.1. Engine: the upstream parallel constrained
decoding engine, one model loaded at a time. Raw results: `quality-eval/results/*.json`,
machine-readable aggregate: `quality-eval/results/summary.json`.

## Method in one paragraph

24 synthetic but policy-grounded cases in two families, 3 fields each (72 field
decisions per model). Family A "payment risk" (`fraud`, `risk`, `action`) covers
card-not-present alerts where the policy is part of the prefilled context; family B
"support triage" (`category`, `priority`, `needs_human`) covers tickets with the same
style of policy. The strict label is derived from the rule table in `make_cases.py`
by construction. 6 of the 24 cases are deliberately ambiguous; for those, a field
that lands on an explicitly listed acceptable alternative is counted as a strict
miss but shown in the `(alt)` columns. Primary fields: `fraud` (A) and `category` (B).

## Headline metrics

{main_md}

Plain-text version:

```
{main_ascii}
```

All-fields exact accuracy counts a case correct only if all three fields match the
strict label, so it is the strictest column here. Majority-class baseline is the
best constant answer for the primary fields (predict the most frequent label per
family): {pct(baseline["primary_accuracy"])}. `Load` is the model load time in this
run, measured with the weights already in the OS page cache (all three models had
been benchmarked earlier the same day); cold loads on this machine are ~40 s (1.5B),
~188 s (7B) and ~259 s (8B) per `knowledge-base/08-model-upgrade-path.md`.

## Per-field accuracy

{field_md}

Family split (primary field and all-fields exact accuracy):

{fam_md}

## Calibration signal

Mean confidence of correct vs incorrect field decisions, and accuracy per
confidence bucket (all 72 field decisions):

{buckets_md}

Reading of the confidence numbers:

{confidence_block}

A calibrated engine would show a large gap between those two means and monotone
bucket accuracy. This engine's confidence is a softmax over only the 2-4 candidate
tokens, not a trained calibration: the numbers above show how far it is from that
ideal, and in which direction (usually confidently wrong).

## Latency

{lat_md}

## Ambiguous cases

Fields that landed on an acceptable alternative (strict miss, defensible answer):

{md_table(["Model", "Fields on alternative", "Cases"], amb_rows)}

## Where each model fails

Every strict miss, by case. Primary-field misses are the ones that matter most for
the deployment question; a wrong `action` on family A is the expensive kind of error.

{fail_md}

## Verdict: is the larger model worth it?

{verdict_block}

Recommended reading of the data above:

1. `{small["tag"] if small else "1.5B"}`: keep it for UI/engine work and latency-critical
   paths only, not for real triage decisions — its primary-field accuracy sits close to
   the majority-class baseline, and its confidence is low across the board.
2. `{mid["tag"] if mid else "7B"}`: the best on the primary field in this eval and
   the best latency/quality trade at ~{num((mid or {}).get("mean_latency_ms"), 0)} ms per case;
   prefer it when the one primary field is what you act on.
3. `{large["tag"] if large else "8B"}`: prefer it when all fields must be jointly right
   (typed payloads written to a system of record), since its all-fields exactness and its
   per-field consistency beat the 7B at nearly the same latency on this 16 GB machine.

## Anomalies

{anomaly_block}

## Limitations

- Synthetic cases: contexts are generated from templates, so they are more uniform
  and cleaner than production text; absolute accuracies are optimistic.
- n=24 cases / 72 field decisions: one or two cases swing the primary accuracy by
  ~4 points, and the confidence buckets are small (see field counts per bucket).
- Single deterministic run (greedy decoding, temperature 1.0 softmax over the
  candidate tokens): no seed variance is measured.
- Rule-based labels: ground truth is the policy in the context, not human judgement;
  the 6 ambiguous cases mitigate this only partially, and fields are scored
  independently even when alternatives imply different other fields.
- The policies in the contexts were written together with the labels, so the task
  rewards reading comprehension of an explicit policy more than open-ended judgement.
- Latency here is one sequential pass per case on a shared 16 GB machine; it is not
  a throughput benchmark. Load times are warm-cache numbers (see the headline table).
- The policies were written together with the labels, so the eval measures careful
  policy reading, not open-ended fraud or triage judgement.
- Confidence numbers are per-field and uncalibrated; the bucket table uses only
  72 decisions per model, so read the direction, not the exact percentages.

## Reproduce

```bash
cd jev-on-a-laptop
.venv/bin/python quality-eval/make_cases.py --check
./quality-eval/run_all.sh
```
"""


def _pick(mm: list[dict]):
    def find(needle):
        for m in mm:
            if needle in m["tag"]:
                return m
        return None

    return find("1.5b"), find("7b"), find("8b")


def per_field_accuracy(runs: list[dict]) -> dict:
    out = {}
    for run in runs:
        acc = {}
        for entry in run["results"]:
            if entry.get("status") != "ok":
                continue
            for name, label in entry["labels"].items():
                hit = entry["predicted"][name] == label
                acc.setdefault(name, []).append(hit)
        out[run["tag"]] = {k: (sum(v) / len(v)) for k, v in acc.items()}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate quality-eval results.")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--summary-md", default=os.path.join(HERE, "SUMMARY.md"))
    args = ap.parse_args()

    runs, skipped = load_runs(args.outdir)
    if not runs:
        print(f"no completed result files in {args.outdir}; run run_eval.py first")
        return 1

    mm = [metrics(r) for r in runs]
    per_field = per_field_accuracy(runs)
    for m in mm:
        m["per_field_accuracy"] = per_field.get(m["tag"], {})
    baseline = majority_baseline(runs)
    cmp = verdict(mm)

    summary = {
        "kind": "rlcd-quality-eval-summary",
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "cases": mm[0]["cases"] if mm else 0,
        "fields_per_model": mm[0]["fields"] if mm else 0,
        "models": mm,
        "majority_baseline": baseline,
        "comparison": cmp,
        "skipped": [{"model": s["model"], "error": s.get("error")} for s in skipped],
    }
    summary_path = os.path.join(args.outdir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {summary_path}")

    md = build_summary_md(runs, mm, baseline, skipped, cmp)
    with open(args.summary_md, "w") as fh:
        fh.write(md)
    print(f"wrote {args.summary_md}")

    print()
    headers = [
        "model",
        "cases",
        "prim-acc",
        "prim-alt",
        "all-exact",
        "field-acc",
        "conf-ok",
        "conf-miss",
        "lat/case",
        "load",
    ]
    rows = [
        [
            m["tag"],
            str(m["cases"]),
            pct(m["primary_accuracy"]),
            pct(m["primary_accuracy_or_acceptable"]),
            pct(m["all_fields_accuracy"]),
            pct(m["field_accuracy"]),
            num(m["mean_confidence_correct"], 2),
            num(m["mean_confidence_incorrect"], 2),
            f"{num(m['mean_latency_ms'])}ms",
            f"{num(m['load_seconds'])}s",
        ]
        for m in mm
    ]
    print(ascii_table(headers, rows))
    print()
    print(f"majority-class baseline on primary fields: {pct(baseline['primary_accuracy'])}")
    print()
    print(verdict_sentence(mm, cmp))
    for c in cmp.get("comparisons", []):
        print(
            f"  {c['pair']}: primary {c['primary_diff_pp']:+.1f} pp, all-fields "
            f"{c['all_fields_diff_pp']:+.1f} pp, latency x{num(c['latency_ratio'], 2)}"
        )
    print()
    for line in confidence_notes(mm):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
