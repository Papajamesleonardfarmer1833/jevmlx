# Changelog

## 0.1.0

- Parallel constrained decisions: every schema field decided in one batched forward pass on Apple Silicon (MLX), with calibrated per-choice probabilities.
- `openjev decide` CLI: run a bundled preset or your own schema/context; table output or `--json`.
- Typed Python API: `openjev.decide(PydanticModel, context)` returns a validated model instance plus per-field confidences.
- Temperature calibration: fit one scalar on labeled JSONL data (`openjev calibrate`) to fix overconfident scores.
- Chat-template handling that works across model families (no hand-built prompts, no system-role assumptions).
- Installable package: `uv pip install .`, presets bundled, `tests/` included (`pip install .[dev]`).
