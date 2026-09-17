# ROADMAP

## Done

- P0: repo hygiene — jevmlx name, README, LICENSE+NOTICE, research-only artifacts dropped.
- P1: installable package — `jevmlx` module, CLI `jevmlx decide`, tokenizer chat templates, smoke test.
- P2: engine quality — consistent scoring for colliding choices, memory auto-chunking.
- P3: typed Python API — `jevmlx.decide(PydanticModel, context)`.
- P4: calibration — `jevmlx calibrate` fits one temperature on labeled JSONL.
- P5: release readiness — model compatibility table, CI, wheel, bundled presets.
- I1: `jevmlx serve` — stdlib HTTP endpoint with health check.
- I2: ruff across the repo (line length 100, E/F/I/UP/B) with CI enforcement.
- I3: ROADMAP.md (this file).
- I4: single-source `__version__`, argument validation hardening.
- I5: GitHub hygiene — build workflow, issue/PR templates, badges.
- C: cleanup pass — only relevant files stay.
- R: professional README restructure.
- W2: documentation and CLI presentation fixes.
- E1-E2, E-baseline, E-metrics, E-report: eval harness (`jevmlx eval` / `jevmlx report`), TypeSafe fetcher, API naive-JSON baseline.

## Next

- W3: correctness gate — token-aligned candidate plans, one scoring objective (token trie), full-precision engine output, strict temperature/memory validation, correct multi-field confidence.
- N1: order-invariance evaluation on the labeled cases, then more labeled data.
- N3: phase-level latency profile (plan compilation, tokenization, prefill, KV replication, suffix pass, scoring).
- N5: PyPI 0.1.0.
- G10: routing thresholds on held-out data (after N1).
- G1: schema-prefix KV reuse in `decide_many` (only if N3 shows it matters).
- N6: Field dependencies: compile schemas into a dependency DAG, decide each independent layer in one batched pass, run deterministic rules between layers, validate cross-field constraints at the end. Today fields are independent one-shot decisions and assembled JSON can be logically inconsistent.
- N7: Typed API exposes full per-field distributions (raw log-scores, alternatives, calibration provenance), not only the winner and one confidence.

## Cut

- Short free-text field type: does not fit the finite-choice guarantee.
- Torch backend: Apple Silicon is the supported platform.

## Rules

- KISS / YAGNI.
- Every phase ends in something runnable.
- Review gate before merge.
