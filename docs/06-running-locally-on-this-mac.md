# 06 — Running it locally on this Mac (verified)

> Machine: **MacBook Air M5, 16 GB unified memory, 8-core GPU, macOS 26.5.1** · Run date 2026-09-16

## What we did

```bash
cd openjev   # repo root
uv venv --python 3.12 .venv                      # system Python is 3.14; 3.12 is the safe target
uv pip install --python .venv/bin/python mlx-lm  # pulls mlx 0.32.2 + mlx-lm 0.31.3

# fetch the artifact
git clone --depth 1 https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD source/Qwen-2.5-1B-RLCD

# run the author's own benchmark (loads mlx-community/Qwen2.5-1.5B-Instruct-4bit)
cd source/Qwen-2.5-1B-RLCD
PYTHONPATH=. ../../.venv/bin/python -u -m core.benchmark
```

Result: it works out of the box. First load (incl. ~1 GB download + Metal warmup): **39 s**; afterwards the engine is warm.

## Measured results — 1.5B-Instruct-4bit on the M5 Air

| Preset | Fields | Naive (autoregressive) | Parallel constrained | Speedup | Naive schema OK? | Parallel schema OK? |
|---|---|---|---|---|---|---|
| FinTech Fraud & AML | 28 | 3261 ms · 288 tok · 88 tok/s | **414 ms** (prefill ~? + suffix ~?) | **7.9x** | ❌ **False** | ✅ True |
| Enterprise Support Triage | 28 | 3312 ms · 283 tok · 85 tok/s | **756 ms** | **4.4x** | ❌ **False** | ✅ True |
| High-Cardinality Tariff (255 choices) | 4 | 888 ms · 53 tok · 60 tok/s | **151 ms** | **5.9x** | ✅ True | ✅ True |

Raw log: `results/raw-1.5b.log`

### What this tells us

- **The speedup is real** and in the same band as the author's claimed 5.6–7.0x (we're on a much weaker chip than the author's M4 Max: 8 GPU cores vs 40, and we still got 4.4–7.9x vs. their baseline).
- **The naive baseline failed schema matching on both 28-field presets** (missing keys / invalid enum values) while the parallel path always matched. That is the single most convincing demonstration of the artifact's premise: a *small model* left to emit raw JSON is unreliable; the same model with constrained parallel decisions is not.
- **Support Triage is ~1.8x slower than Fintech despite the same field count** — almost certainly the collision-continuation path (see `05`, step 7), where fields whose choices share a first token force extra sequential passes. This is a concrete optimization target.
- Absolute times (414–756 ms for 28 fields) are ~1.5–2.8x slower than the author's M4 Max numbers (270 ms) — expected for an Air-class GPU.

**Larger models:** the same benchmark on `Qwen2.5-7B-Instruct-4bit` also ran successfully on this machine (28 fields: 1.5–1.7 s parallel, 7.0–7.9x speedup, naive schema still unreliable). Full comparison in `08` (including Qwen3-8B).

## Running other models (our runner)

The artifact hardcodes `MODEL_ID` in `core/engine_mlx.py`, but it reads the module global at load time, so we can override it without editing their code:

```bash
cd openjev   # repo root
.venv/bin/python -u tools/bench_model.py mlx-community/Qwen2.5-7B-Instruct-4bit
# or: --tag my-label  (results land in results/<tag>.json)
```

## Optional: the side-by-side web UI

```bash
cd rlcd-research
uv pip install --python .venv/bin/python fastapi uvicorn pydantic
cd source/Qwen-2.5-1B-RLCD && PYTHONPATH=. ../../.venv/bin/python -m uvicorn server.app:app --port 8000
# open http://localhost:8000
```

## Gotchas encountered

- System Python 3.14 → use 3.12 for the venv (mlx-lm wheels + numpy are safest there).
- First HF download prints an unauthenticated-requests warning; harmless.
- The engine's Metal warmup repeats a 28-wide broadcast once before the first real call — that's why the very first run has a warmup phase.
- Peak memory is modest for 1.5B-4bit (~1.1 GB weights + broadcast KV), but scales with **fields × context length**; see `07` for numbers.
