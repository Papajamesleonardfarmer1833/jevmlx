# Changelog

## Unreleased

- Renamed openjev -> jevmlx (package, CLI command, env var `JEVMLX_LOG`, cache dir `~/.cache/jevmlx`, repo URL).
- Trie scoring merged (W3): each field is a token trie over its choices; at each branch point the model's next-token distribution is restricted to the allowed tokens, and P(choice) is the product of those branch probabilities — so the field's probabilities sum to 1 with no extra softmax.
- Eval harness merged: `jevmlx eval` runs labeled cases through the parallel, naive-local, or API-baseline tracks and writes per-field predictions plus a run manifest (`predictions.jsonl` / `run.json`); `jevmlx report` summarizes runs offline.
- Correctness gate in progress: schema candidates are tokenized as complete sequences (token-aligned plans, fixing character-prefix splitting), the engine scores all fields with one objective (a token trie over choice remainders), engine outputs keep full precision, temperature and memory-batch arguments are validated, and multi-field confidence is the minimum over all per-option decisions.
- CLI presentation: confidences are rounded only at display time (3 decimals in the table, 4 in `--json`), never in the engine.
- Documentation: README and ROADMAP claims aligned with the code (chunked suffix passes, tested-model boundary, no memory-bound guarantee).
- Prompt v2 and a new default scoring mode: the prompt uses neutral aliases (A, B, C) for the choices and the engine scores quoted-alias candidates through the token trie; the original choice-text mode is still available as `--scoring labels`.
- Opt-in prior correction (`--prior-correction`): the engine measures each field's choice distribution on a neutral context once per schema and subtracts that prior from the evidence scores before selecting; off by default (it lowered accuracy on the 0.5B bundled set).
- Per-field provenance in the Python API: every decision exposes its probability, score, margin, the top alternatives, whether a calibration has been applied, and which scoring mode produced it; optional per-choice glosses can be attached to the schema, and Optional enum fields can accept an explicit "insufficient evidence" answer mapped to None.
- Multi-select fields are decided as separate yes/no rows with a documented threshold, instead of one ranked list.
- Synthetic labeled cases covering the known failure modes ship with the benchmark data.
- New backend: the same decision semantics run through any OpenAI-compatible chat endpoint that returns logprobs (`jevmlx decide --backend openai`), one request per field; missing candidates in the endpoint's top-k get an explicit floor probability and are flagged in the output.
- New `jevmlx doctor` command: checks the environment (Python, MLX, Metal, model cache) before you file an issue or run a benchmark.
- New `jevmlx bench` command: one command that produces a complete, PR-ready results folder (compatibility table inputs, eval reports, environment metadata).
- BENCHMARKING.md: how to run and contribute benchmark results from your own machine.

## 0.1.0

- Parallel constrained decisions: every schema field decided in one batched forward pass on Apple Silicon (MLX), with per-choice probabilities from softmax over choice scores (see Calibration for fitting a temperature).
- `jevmlx decide` CLI: run a bundled preset or your own schema/context; table output or `--json`.
- Typed Python API: `jevmlx.decide(PydanticModel, context)` returns a validated model instance plus per-field confidences.
- Temperature calibration: fit one scalar on labeled JSONL data (`jevmlx calibrate`) to fix overconfident scores.
- Chat-template handling that works across model families (no hand-built prompts, no system-role assumptions).
- Installable package: `uv pip install .`, presets bundled, `tests/` included (`pip install .[dev]`).
