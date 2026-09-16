# 08 — Model upgrade path: 1.5B → 7B → 8B on this 16 GB Mac

> The artifact's engine takes any `mlx-lm` model. We test successively larger models with the same three presets, on the same Mac (M5, 16 GB), to find the largest model that *fits and stays useful*.

## How any model gets swapped in

No code edits needed — `core/engine_mlx.py` reads the module global `MODEL_ID` at load time, so our runner overrides it:

```bash
cd rlcd-research
.venv/bin/python -u tools/bench_model.py <hf-model-id> [--tag label]
```

Ours: `tools/bench_model.py`. It runs the same naive-vs-parallel comparison as `core/benchmark.py` and saves JSON to `results/`.

## Candidates & fit on 16 GB unified memory

| Model | Weights (4-bit) | Est. peak (28-field broadcast, ~1k-token context) | Verdict |
|---|---|---|---|
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | ~1.1 GB | ~2 GB | ✅ comfortable — the reference run |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | ~4.3 GB | ~6 GB | ✅ fits, ~30 tok/s class decode |
| `mlx-community/Qwen3-8B-4bit` | ~4.8 GB | ~9 GB | ⚠️ fits, tighter; Qwen3 KV is larger (8 KV heads) |
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | ~5–6 GB | ~10 GB+ | ⚠️ risky: VL-family arch (`Qwen3_5ForConditionalGeneration`), bigger KV, prompt template mismatch — test only if curious |
| `mlx-community/Qwen2.5-14B-Instruct-4bit` | ~8 GB | ~11 GB+ | ❌ not worth it on 16 GB with the broadcast; swapping/thermal risk |

> Rule of thumb: with this engine, peak ≈ weights + fields × KV(context). Fields multiply the *cache*, not the weights — so context length is as dangerous as model size.

## Results

### 1.5B — done (`results/raw-1.5b.log`)

| Preset | Fields | Naive | Parallel | Speedup | Naive schema | Parallel schema |
|---|---|---|---|---|---|---|
| FinTech (28) | 28 | 3261 ms | **414 ms** | 7.9x | ❌ | ✅ |
| Support Triage (28) | 28 | 3312 ms | **756 ms** | 4.4x | ❌ | ✅ |
| High-Cardinality 255 (4) | 4 | 888 ms | **151 ms** | 5.9x | ✅ | ✅ |

### 7B — ✅ done (`results/raw-7b.log`)

Load time (first run, incl. ~4.3 GB download): 188 s.

| Preset | Fields | Naive | Parallel (prefill + suffix) | Speedup | Naive schema | Parallel schema |
|---|---|---|---|---|---|---|
| FinTech (28) | 28 | 11916 ms · 298 tok · 25.0 tok/s | **1518 ms** (863 + 553) | 7.9x | ❌ | ✅ |
| Support Triage (28) | 28 | 12144 ms · 296 tok · 24.4 tok/s | **1735 ms** (884 + 553) | 7.0x | ✅ | ✅ |
| High-Cardinality 255 (4) | 4 | 2659 ms · 49 tok · 18.4 tok/s | **790 ms** (445 + 198) | 3.4x | ❌ | ✅ |

**What changed vs. 1.5B:**

- Parallel latency grew **2.3–5.2x** (fintech 3.7x, triage 2.3x, tariff 5.2x) — as expected from the model size alone. It fits comfortably in 16 GB: no memory pressure, no swap, no OOM.
- **Prefill now dominates** the 28-field parallel path: 863 of 1518 ms (57%) for fintech. The one-time context read is the main cost on the larger model; the batched suffix pass is **identical (553 ms) across both 28-field presets**, showing field-batch cost is stable and predictable.
- The naive baseline is **~3.7x slower** than at 1.5B (11916 vs 3261 ms) → the speedup *vs naive* stays high (7.9x / 7.0x) even though absolute parallel latency tripled.
- **Naive JSON is still unreliable at 7B**: failed schema on fintech and tariff (only triage passed). A 7B *still* cannot be trusted to emit 28-field JSON unconstrained.
- Unmetered remainder (elapsed − prefill − suffix): ~102 ms fintech vs ~298 ms triage → consistent with the collision-continuation cost hypothesis (`05`, sharp edge #2).

### Qwen3-8B — ✅ done (`results/raw-qwen3-8b.log`)

Load time (first run, incl. download): 259 s.

| Preset | Fields | Naive | Parallel (prefill + suffix) | Speedup | Naive schema | Parallel schema |
|---|---|---|---|---|---|---|
| FinTech (28) | 28 | 14279 ms · 302 tok · 21.1 tok/s | **2031 ms** (1005 + 883) | 7.0x | ❌ | ✅ |
| Support Triage (28) | 28 | 14707 ms · 307 tok · 20.9 tok/s | **2053 ms** (1033 + 658) | 7.2x | ❌ | ✅ |
| High-Cardinality 255 (4) | 4 | 3484 ms · 54 tok · 15.5 tok/s | **965 ms** (539 + 247) | 3.6x | ✅ | ✅ |

**Verdict after three models:**

- Latency grows with size: 28-field parallel is **0.41 s (1.5B) → 1.52 s (7B) → 2.03 s (8B)**. Prefill dominates for 28 fields (8B: 1005 of 2031 ms).
- Qwen3-8B is **not clearly better than 7B for this workload** — it's ~33% slower, and it additionally failed the naive-JSON sanity check on the triage preset. It does fit, though: no OOM, no swap.
- Decision on size is therefore not "biggest that fits" but "**7B is the sweet spot** on this 16 GB Mac; 8B only if a quality eval shows it decisively better" — see `quality-eval/` (built 2026-09-16) for the accuracy comparison.

If the 7B fits, Qwen3-8B is the more interesting ceiling test: newer generation, likely the best decision quality we can get locally in this size class. Caveat: Qwen3 expects `enable_thinking=False` handling in its chat template, but this engine hardcodes raw ChatML (no think block) — usually fine for non-thinking use, verify empirically.

## Interpreting the results (analysis slots)

Fill in once data is in:
- **Latency ratio** 7B/1.5B (parallel), and whether the *speedup vs naive* holds (bigger models generate slower → naive gets worse → speedup can grow).
- **Schema validity** of the naive baseline at 7B (does a better model fix its JSON? usually yes, partially).
- **Decision quality** — note that these presets have no ground truth; until Experiment A1 (`09`) exists, we can only compare *agreement between models* as a proxy.

## Recommendation ladder

1. **1.5B** — reference implementation; fast; good for engine/UI work.
2. **7B-4bit** — the pragmatic default for actual decision quality on this Mac, if latency in the ~1–2 s range is acceptable per call.
3. **Qwen3-8B-4bit** — try as the top of the practical range.
4. Scale beyond 8B → switch hosts or go to training/adapters (`07`, `09`), not bigger zero-shot models.
