# ROADMAP

## Done

| Phase | Deliverable | Done |
|---|---|---|
| P0 | Repo hygiene: openjev name, README, LICENSE+NOTICE, drop x-posts/hf-space/vendor | 956f11d |
| P1 | Package: `openjev/` module, pyproject, `uv pip install -e .`, CLI `openjev decide`, chat template from tokenizer, smoke test | 67a2f80 |
| P2 | Engine: multi-token collision scoring in one pass, honest confidence, memory auto-chunk | bcf7519 |
| P3 | API: Pydantic model in, typed object out | 1e1cf6d |
| P4 | Calibration: `openjev calibrate` fits temperature on labeled JSONL | 75afe53 |
| P5 | Public: model compatibility table, CI, PyPI | 07be0ed |

## Next

| Phase | Deliverable |
|---|---|
| N3 | Latency: 860 ms here vs 410 ms upstream for 1.5B/fintech_fraud, unexplained. Profile prefill on a quiet machine. |
| N1 | Accuracy: eval loop on quality-eval + evals data, then prompt/description work. 7B is 73.8% vs Jev 86.6%. |
| N4 | `openjev serve` HTTP endpoint (in progress, I1). |
| N2 | More field types: int range, date, short text. Only on request. |
| N5 | Publish: PyPI, GitHub release. |

## Rules

- KISS / YAGNI.
- Every phase ends in something runnable.
- Review gate before merge.
