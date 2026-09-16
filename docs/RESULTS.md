# Results — all measured runs

Machine: **MacBook Air M5, 16 GB unified memory, macOS 26.5.1** · mlx 0.32.2 / mlx-lm 0.31.3 · dates 2026-09-16

Runner: `tools/bench_model.py`. Raw JSON in `results/`. "Naive" = same model generating the full JSON string autoregressively (greedy). "Parallel" = one batched constrained pass.

## 1. Basic latency table

| Preset | Fields | Model | Naive | Parallel | Speedup | Naive schema | Parallel schema |
|---|---|---|---|---|---|---|---|
| FinTech Fraud | 28 | 1.5B | 3261 ms | 414 ms | 7.9x | ❌ | ✅ |
| FinTech Fraud | 28 | 7B | 11916 ms | 1518 ms | 7.9x | ❌ | ✅ |
| FinTech Fraud | 28 | 8B | 14279 ms | 2031 ms | 7.0x | ❌ | ✅ |
| Support Triage | 28 | 1.5B | 3312 ms | 756 ms | 4.4x | ❌ | ✅ |
| Support Triage | 28 | 7B | 12144 ms | 1735 ms | 7.0x | ✅ | ✅ |
| Support Triage | 28 | 8B | 14707 ms | 2053 ms | 7.2x | ❌ | ✅ |
| Tariff (255 choices) | 4 | 1.5B | 888 ms | 151 ms | 5.9x | ✅ | ✅ |
| Tariff (255 choices) | 4 | 7B | 2659 ms | 790 ms | 3.4x | ❌ | ✅ |
| Tariff (255 choices) | 4 | 8B | 3484 ms | 965 ms | 3.6x | ✅ | ✅ |

Notes:

- Parallel latency breakdown (prefill + batched pass): 1.5B 414 ms unmeasured-split; 7B 863+553; 8B 1005+883 (fintech). **Prefill dominates at 28 fields.**
- Support triage is consistently slower than fintech at equal field count → the collision-continuation path (see `docs/05`).
- All three parallel runs were schema-valid in every preset. Naive generation failed schema validity in 5 of 9 runs.

## 2. Model sizing on 16 GB (verdict)

| Model | Fits? | 28-field latency | Verdict |
|---|---|---|---|
| Qwen2.5-1.5B-4bit | ✅ comfy | ~0.4–0.8 s | fastest; good for engine/UI work |
| Qwen2.5-7B-4bit | ✅ comfy | ~1.5–1.7 s | **sweet spot** for decision quality/latency |
| Qwen3-8B-4bit | ⚠️ fits | ~2.0 s | no clear win over 7B in our runs |
| 14B-4bit | ❌ risky | — | not recommended on 16 GB with the broadcast |

## 3. Decision quality (labeled eval)

See `../quality-eval/SUMMARY.md` (generated 2026-09-16) for per-model accuracy, calibration buckets, and the explicit "is bigger worth it" verdict on synthetic labeled cases.
