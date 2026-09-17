# benchmarks/results/

One directory per eval run, written by `jevmlx eval` / `jevmlx report`
(to be populated by the N1 run):

```
<run_id>/
  predictions.jsonl   one JSON object per field decision (see the eval contract)
  run.json            run manifest: model, tracks, cases, timestamps
  report.json         metrics computed by `jevmlx report`
  report.md           human-readable summary of report.json
```
