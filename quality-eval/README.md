# quality-eval — does a larger model make better decisions?

The artifact's benchmark (`source/Qwen-2.5-1B-RLCD`) measures **speed only**. This
folder measures **decision quality** on a small labeled set, so "is the bigger model
worth it?" has an answer with numbers behind it.

## What is measured

24 deterministic, rule-labeled cases in two families, 3 fields each (72 field
decisions per model):

| Family | Cases | Schema | Primary field |
|---|---|---|---|
| A — payment risk | 12 | `fraud` bool, `risk` LOW/ELEVATED/HIGH/CRITICAL, `action` APPROVE/REVIEW/BLOCK | `fraud` |
| B — support triage | 12 | `category` BILLING/TECHNICAL/ACCOUNT/SHIPPING, `priority` P1/P2/P3, `needs_human` bool | `category` |

Each context contains an explicit policy plus an alert or ticket. Ground truth comes
from the rule tables in `make_cases.py` (documented in its module docstring), not
from hand annotation. 6 cases (A10-A12, B10-B12) are deliberately ambiguous and
carry an explicit acceptable-alternative set; a strict miss inside that set is still
a strict miss but is reported in the `(alt)` columns.

## Files

```
quality-eval/
├── make_cases.py     generates cases.json (deterministic; --check verifies freshness)
├── cases.json        the 24 cases (schema, context, labels, acceptable alternatives)
├── run_eval.py       CLI: --model <hf-id> [--limit N --tag X --artifact P --outdir D]
├── analyze.py        merges results/*.json -> results/summary.json + SUMMARY.md
├── run_all.sh        runs the 3 cached models sequentially, then analyze
├── results/          per-model JSON, summary.json, run_all.log
├── results-smoke/    smoke runs, ignored by analyze.py (tag contains "smoke")
└── SUMMARY.md        the generated report (verdict + tables)
```

## Run it

```bash
cd rlcd-research
.venv/bin/python quality-eval/make_cases.py --check   # confirm cases.json is current
./quality-eval/run_all.sh                             # all three models, then analyze

# single model, or a fast smoke test
.venv/bin/python quality-eval/run_eval.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
LIMIT=4 ./quality-eval/run_all.sh
```

`run_all.sh` is sequential on purpose: 16 GB of unified memory fits one model at a
time. It never downloads; a model missing from the Hugging Face cache is skipped
(with a note) and `run_eval.py` writes a `status=load_failed` result instead of
crashing the driver.

## Metrics

- **Primary-field accuracy** — `fraud` (A) and `category` (B).
- **All-fields exact accuracy** — all three fields of a case correct.
- **Acceptable-inclusive accuracy** — same, counted leniently on the ambiguous cases.
- **Calibration signal** — mean confidence on correct vs incorrect field decisions,
  plus accuracy per confidence bucket (<0.70 / 0.70-0.90 / >0.90).
- **Latency** — mean/median `elapsed_ms` per case, mean prefill vs suffix, load time.
- **Ambiguity count** — fields that land on an acceptable alternative.

## Engine facts worth remembering when reading the numbers

- The engine's system prompt lists field **names and descriptions only** — the
  allowed choices are not shown to the model. `model.py`/`schema.py` in `source/`
  is untouched (read-only, by policy), so this is measured as-is.
- Enum choices were chosen so no two first tokens collide (verified with the Qwen2.5
  tokenizer): collisions would route a field through the engine's slower,
  fabricated-confidence continuation path.
- The reported confidence is a softmax over 2-4 candidate tokens, not a trained
  calibration. Treat it as "veto strength", not as a probability you can act on.

## Limitations

Synthetic template contexts, n=24, single deterministic run per model, rule-based
labels, and policies authored together with the labels. See the Limitations section
of `SUMMARY.md`. Do not quote the absolute accuracies as production expectations —
quote the **relative** ordering of the models.
