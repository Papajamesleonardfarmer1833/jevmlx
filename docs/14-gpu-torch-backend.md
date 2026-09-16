# 13 — The GPU update: porting the upstream torch backend, and what CUDA actually buys

**Date:** 2026-09-16 · **Machine:** the "PC" from doc 07 (RTX 2060 SUPER, 8 GB VRAM, 32 GB RAM)
**Upstream:** [`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) commit `031d1a8` (2026-09-16 06:15 UTC, "Add live ZeroGPU Space reference and dual MLX/PyTorch engine support")

## What actually shipped upstream today

The "70× speedup" that prompted this session is not what the commit contains.
The model card still claims **5.6×–7.0×** vs *autoregressive* decoding (M4 Max).
What `031d1a8` really adds is a **PyTorch backend** (`core/engine_torch.py`):
the same parallel-constrained decoding, executable on CUDA GPUs and CPU — built
so the demo can run on HF Spaces ZeroGPU. MLX remains the Apple-Silicon engine.

## What we did with it

`parallel-decisions` (the release package) previously **refused to run** on
non-Apple-Silicon machines (`UnsupportedPlatformError`). We integrated the torch
backend *into the package* rather than vendoring it: the upstream port scores
first-token logits only — no collision resolution, no multi-select, no chunking,
no calibration. Ours reuses the package's schema compiler, collision scorer,
multi assembly and calibrator, so the torch path has feature parity with MLX.

- `Decider(backend="auto"|"mlx"|"torch")`, `pd.toml backend=...`, `PD_BACKEND`;
  `auto` = mlx on Apple Silicon, torch (CUDA when available) elsewhere.
- `torch_dtype` / `torch_device` tuning; weights in bf16 on CUDA.
- Windows fix: `_physical_ram_bytes()` via `GlobalMemoryStatusEx` (sysconf has no
  memory keys on Windows), so the memory clamp works here too.
- 131 tests pass (3 macOS-only skips); new regression tests for the two bugs below.

## Two real bugs the browser lab caught (and the benchmarks missed)

1. **Broadcast corrupted the prompt cache.** HF's
   `DynamicCache.batch_repeat_interleave` mutates layer objects in place; a
   shallow cache copy still *shares* those layers. The first broadcast silently
   rewrote the caller's cache, so any second pass over it — collision rows,
   chunking, prefix reuse — crashed in `torch.cat` (`Expected size 15 but got
   size 5`). Benchmark schemas were collision-free and single-chunk, so they
   never exercised the second pass. Fix: broadcast copies each layer object and
   writes fresh repeated tensors; the original is never touched.
2. **Missing `import torch`** in the collision softmax path — NameError on the
   first colliding decision.

The email-triage schema (`ARCHIVE/REVIEW/BLOCK_SENDER/PAY/REPLY`, plus
`PHISHING/FRAUD/...`) has collisions, so the lab hit both on day one. This is
the E1 lesson: the cross-check is not a formality.

## Numbers

Per-request decision latency, `Qwen2.5-0.5B-Instruct` bf16, prompt ≈ 816 tokens,
3-field schema (10 decision rows incl. one collision pass), 20 runs:

| Configuration | p50 latency | Throughput |
|---|---|---|
| CPU (fp32) | 2 887 ms | 0.35 req/s |
| **CUDA serial** | **~355 ms** | **~2.8 req/s** |
| CUDA ×2 processes | ~990 ms | 1.48 req/s |
| CUDA ×4 processes | ~3 900 ms | 0.39 req/s (thrash) |

**CUDA is 8.1× faster than CPU on this card.** The short-context presets from
doc 07 (~100 tokens) run at **~109–123 ms** per decision on CUDA.

End-to-end **browser actions** (local email-triage page, 10 requests fired
concurrently from real Chrome via playwright-cli, server-side model in the loop):

| Model processes | Wall for 10 actions | Per-action (server) |
|---|---|---|
| 1 | **2 007 / 1 980 ms** | ~190 ms each |
| 2 | 2 497 ms | ~465–1 570 ms |

## What this means for "parallelizing many browser actions"

- The engine is **already parallel internally**: one prefill + one batched pass
  answers all 10 decision rows at once. Per-request latency is ~190 ms and is
  dominated by the prefill of an 816-token prompt, not by field count.
- **Don't shard by process on an 8 GB card.** Two 0.5B models already contend
  for SMs; ×4 thrashes. The engine's own batching is the concurrency.
- The lever that *would* help is **prefix reuse across requests** (the schema
  block is identical every time): `prepare()`/`decide_with_prefix()` exist in the
  MLX engine and are the next port for the torch path — worth ~3–4× on prefill.
- Bigger models: 7B bf16 needs ~15 GB (weights + broadcast KV) — does not fit
  8 GB VRAM; doc 07's prediction stands. The 1.5B–3B class is the card's sweet
  spot; 0.5B runs at ~0.36 s/decision.

## Where the code lives

- `parallel-decisions` (release): `src/parallel_decisions/engine_torch.py`,
  backend selection in `engine.py`/`config.py`, tests in
  `tests/test_engine_torch.py`. Merged to `main`, tags pending release.
- `jev-on-a-laptop` (mirror): `evals/bench_torch_backends.py` (the table above),
  `evals/browser_lab/` (the toy app + page used with playwright-cli).
- Upstream origin: `rlcd-upstream` clone, commit `031d1a8`; mirrored raw files
  under `docs/` provenance. (The upstream repo is HF-hosted, not GitHub.)

## Reproduce

```bash
# decisions on GPU
.venv/Scripts/python -c "from parallel_decisions import Decider, Schema; ..."
# benchmark matrix
.venv/Scripts/python evals/bench_torch_backends.py --device cuda --runs 20 --cpu-baseline
# browser lab
.venv/Scripts/python evals/browser_lab/app.py --port 8766 --workers 1
# then: playwright-cli -s=chrome goto http://127.0.0.1:8766/
```

