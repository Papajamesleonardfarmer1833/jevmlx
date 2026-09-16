# 09 — Open questions & the experiment plan

> This is the "what's next" doc. Once each item is done, its result moves into `06`/`08` or a new numbered doc.

## A. Decision quality (the thing the artifact does NOT benchmark)

The repo only measures speed and schema validity. It never asks: **are the chosen values correct?** That's the gap that matters if we ever want to trust it.

**Experiment A1 — build a labeled decision eval.**
- Take the 4 presets (they ship realistic contexts) and hand-label the correct answers per field (28+28+28+4 fields = 88 decisions; can start with a 20-field subset).
- Include deliberately ambiguous cases to see how the model behaves.
- Metric: per-field accuracy + full-schema-exact accuracy, for 1.5B vs 7B vs Qwen3-8B.

**Experiment A2 — collision audit.**
- Instrument the engine to report which fields take the collision path. Hypothesis: support-triage's extra ~340 ms comes from specific choice sets.
- Fix candidates: rebuild choices to avoid shared first tokens where possible; batched trie disambiguation; or merge continuation tokens into the candidate set.

**Experiment A3 — calibration check.**
- Bucket `prob` values from the non-collision path into bins (0.5–0.6, …, 0.9–1.0) and plot accuracy per bin (reliability diagram).
- This directly tests the artifact's "calibrated" claim — expect it to be *miscalibrated* (raw softmax at T=1 usually is). If so, the cheapest honest improvement is temperature scaling on a held-out set.

## B. Beyond booleans & enums

TypeSafe has **Score** (rubric → number + distribution) and **Noul** (statement → 0–1). The artifact has no equivalent.

- **B1 — numeric fields** as a constrained set of bins or as token-slice scoring over 0–9 / 00–99 (easy, keeps the same machinery).
- **B2 — free short strings** via a trie over a known vocabulary (e.g. category names not known up front).
- **B3 — true "score" primitive**: put a numeric head over the logits of digit tokens and normalize — a weekend-scale extension of `extract_calibrated_probabilities`.

## C. Training our own (if we want real RLCD-ish behavior)

Not possible to reproduce TypeSafe's method (undocumented), but a **reasonable local stand-in**:
1. Generate synthetic preference data: same state → multiple candidate decision sets, label "better" by a stronger model's judgment (we have frontier-ish local models; or use the OpenRouter API if a key is available).
2. Train a small adapter (LoRA) on the decision task with a calibration-aware loss (e.g. soft-label cross-entropy against a stronger model's *probabilities*), not just accuracy.
3. Evaluate with A1/A3.

Host: **the PC** if we do torch/PEFT, or MLX LoRA on the Mac (1.5B–3B only). This is the natural "use the PC" project.

## D. Practical engineering tasks (cheap, high value)

- **D1 — Surface the collision flag** in `field_telemetry` and in the web UI.
- **D2 — Batch collision resolution**: replace per-field sequential loops with a token-tree pass.
- **D3 — Batch multiple *states* in one call** (same schema, many inputs). The current code evaluates one state; batching states is how you'd actually feed a pipeline (map-reduce over data — one of TypeSafe's stated use cases). This also amortizes prefill.
- **D4 — Clean up the boolean inconsistency** (`parsed_json` uses real booleans, telemetry uses strings).
- **D5 — Sampling/temperature policy**: allow `temperature < 1` for sharper confidence without changing argmax; decide a policy and document it.

## E. Cross-machine validation

- **E1 — run `core/engine_torch.py` on the PC** with the same presets (recipe in `07`). Question: does CUDA change the ranking, and does 7B 4-bit OOM at 8 GB VRAM?
- **E2 — measure the Mac under sustained load** (e.g. 200 consecutive parallel calls) to expose fanless throttling.

## F. Written deliverables we might want later

- A short technical write-up ("We ran a Jev-style decoder on a stock Qwen model on a laptop: here's what's real and what's marketing") — the KB already has the raw material.
- A minimal, cleaned-up Python package of the good parts of the engine (`parallel_decisions`) with the fixes above, if we decide to build on it.

## Status tracker

| Item | Status |
|---|---|
| Run 1.5B on M5 | ✅ done (`06`) — 4.4–7.9x, naive fails schema on 28-field presets |
| Run 7B on M5 | ✅ done (`08`) — 1.5–1.7 s parallel, 7.0–7.9x, fits 16 GB fine |
| Run Qwen3-8B on M5 | ✅ done (`08`) — 2.0 s parallel, fits, no clear win over 7B |
| A1 labeled eval | ⏳ not started |
| A2 collision audit | ⏳ hypothesis formed (`05`), not measured |
| A3 calibration | ⏳ not started |
| E1 PC cross-check | ⏳ script ready in `07`; needs the actual PC |
