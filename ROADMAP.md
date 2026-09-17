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
- W3: correctness gate merged — token-aligned candidate plans, one scoring objective (token trie), full-precision engine output, strict temperature/memory validation, correct multi-field confidence.
- X: repo rename (openjev -> jevmlx) and README claims cleanup.
- V1: prompt v2 + slot-trie default scoring (neutral aliases in the prompt, quoted-alias candidates; labels mode remains opt-in via `--scoring labels`).
- V2: opt-in prior correction (`--prior-correction`): the engine measures each field's choice distribution on a neutral context once per schema and subtracts that prior from the evidence scores before selecting; off by default (it lowered accuracy on the 0.5B bundled set).
- V3: provenance API — per-field `Decision.fields` (probability, score, margin, alternatives, calibration status, scoring mode), per-choice glosses, and `allow_unknown` mapping an UNKNOWN choice to None for Optional fields.
- V4: multi fields as natural yes/no rows — described options, Y/N aliases, exposed threshold.
- V5: synthetic labeled cases for the named failure modes, feeding the eval harness.
- A1: OpenAI-compatible backend — the same slots decision semantics through any logprobs-capable chat endpoint (`jevmlx decide --backend openai`, `openai_slots` eval track), one request per field, top-k renormalisation with explicit truncation flags.
- D1: `jevmlx doctor` — environment checks before an issue report or a benchmark run.
- B1: `jevmlx bench` — one command to a complete PR-ready results folder.
- E-metrics: TypeSafe-comparable metrics — agreement, TVD vs consensus, report table.
- Docs: BENCHMARKING.md contributing guide, README user guide + table of contents, ARCHITECTURE.md + CONTRIBUTING refresh.

## Next

- N1: order-invariance evaluation on the labeled cases, then more labeled data.
- N3: phase-level latency profile (plan compilation, tokenization, prefill, KV replication, suffix pass, scoring).
- M2: bigger models on Apple Silicon — Qwen 3.8 27B, Qwen Next Flash, Gemma 4 on an M5 Max; GLM via API as baseline.
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
