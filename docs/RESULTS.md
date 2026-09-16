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

`quality-eval/` — 24 synthetic, policy-grounded cases (12 payment risk, 12 support triage, 3 fields each, 6 deliberately ambiguous). Primary field = the one you'd act on (`fraud` / `category`).

| Model | Primary acc | All-fields exact | Mean conf (correct) | Mean conf (wrong) | Latency/case |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 58.3% | 50.0% | 0.73 | 0.60 | 147 ms |
| Qwen2.5-7B | **95.8%** | 72.2% | 0.96 | 0.90 | 611 ms |
| Qwen3-8B | 91.7% | **84.7%** | 0.89 | 0.84 | 646 ms |
| majority-class baseline | 54.2% | — | — | — | — |

**Verdict:** the 1.5B sits at the majority-class baseline (not usable for real decisions); 7B is the best primary-field model at the best latency/quality trade (use when one field is what you act on); 8B wins on all-fields exactness (use when the whole typed payload must be right). **Confidence is a weak error signal:** the 7B was >0.90 confident on 13 of its 20 wrong fields — do not threshold on it without further calibration work.

Details, per-field breakdown, failure analysis, and limitations: `quality-eval/SUMMARY.md`.

## 4. How this compares to Jev's published accuracy

TypeSafe's own workflow evals (https://evals.typesafe.ai/) put Jev at **67.8% mean accuracy** (61.7–76.0% per workflow) against a **frontier consensus** (average of GPT-6 Astra and Fable 5.1 answering every question), at $0.0004 and 0.4 s per case. Our 8B's 84.7%/91.7% is **not comparable** (rule-constructed labels, synthetic cases, n=24) — it would imply beating Opus 5 (73.1%) and Sol (74.1%) on their eval, which is implausible. Comparable findings: same-order latency (0.4 s vs 0.65 s) and a ~100–1000x local cost advantage, both type-safe by construction. Full analysis in `docs/10-jev-published-accuracy-vs-our-8b.md`.
