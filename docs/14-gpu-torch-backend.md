# 14 — The GPU update: porting the upstream torch backend, and what CUDA actually buys

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
  memory keys on Windows). This probe does **not** mean Torch enforces an adaptive
  VRAM budget: its chunk size is currently `max_fields_per_batch`.
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

End-to-end **browser-triggered classification requests** (local email-triage page;
10 requests fired concurrently from real Chrome via playwright-cli, server-side
model in the loop. Note: the page then re-renders recommendation badges locally —
this measures model-backed HTTP round trips, not ten model-driven browser clicks):

| Model processes | Wall for 10 requests | Per-request (server) |
|---|---|---|
| 1 | **2 007 / 1 980 ms** | ~190 ms each |
| 2 | 2 497 ms | ~465–1 570 ms |

## What this means for "parallelizing many browser actions"

- The engine is **already parallel internally**: it reuses one prefill for
  batched decision rows, with extra passes for collisions/chunks. The lab's
  ~190 ms server timing is not interchangeable with the 816-token benchmark.
  Field count, prompt length and collisions all affect latency.
- **Don't shard by process on an 8 GB card.** Two 0.5B models already contend
  for SMs; ×4 thrashes. The engine's own batching is the concurrency.
- The lever that *would* help is **prefix reuse across requests** (the schema
  block is identical every time): `prepare()`/`decide_with_prefix()` exist in the
  MLX engine and are the next port for the torch path — then projected at ~3–4×
  on prefill; **measured 2026-09-17, see the addendum: 1.87× on prefill,
  1.43× wall for the measured workload.**
- Bigger models: 7B fp16/bf16 weights alone need roughly 14 GB — do not fit
  8 GB VRAM. Start with the measured 0.5B model; larger models need their own
  memory/latency evaluation. Desktop GPU use further reduces available memory.

## Update: measured prefix reuse and CUDA Graphs (2026-09-17)

**Setup and how-to:** [GPU_SETUP.md in the release library](https://github.com/rorshopping/parallel-decisions/blob/main/GPU_SETUP.md).
This is CUDA **inference**, not video rendering or GPU-accelerated browser clicks.

All rows below use Qwen2.5-0.5B-Instruct fp16 on the RTX 2060 SUPER. Each row is a
separate paired experiment; they do not use the earlier CPU/CUDA workload.

| Optimization | Request latency before → after | Gain | Sample |
|---|---|---|---|
| Shared schema prefix | 88.5 → 61.9 ms, means | 1.43×; 30% less latency | 8 fields, 405 prefix tokens, 10 pairs |
| CUDA Graph replay | 60.1 → 51.8 ms, medians | 1.16×; 14% less latency | 8 synthetic boolean fields, 10 pairs |
| CUDA Graph replay | 240.9 → 83.8 ms, medians | 2.88×; 65% less latency | Custom 27-field schema, 6 pairs |

Raw evidence in the release repository's `benchmarks/`:
[`prefix-results.json`](https://github.com/rorshopping/parallel-decisions/blob/main/benchmarks/prefix-results.json),
[`cuda-graph-results.json`](https://github.com/rorshopping/parallel-decisions/blob/main/benchmarks/cuda-graph-results.json),
[`cuda-graph-27-results.json`](https://github.com/rorshopping/parallel-decisions/blob/main/benchmarks/cuda-graph-27-results.json).
The custom schema is not the historical 28-field probe. All selected answers
matched within each paired experiment; graph runs recorded zero probability error.
This establishes consistency for these inputs, not task accuracy.

- Prefix preparation cost ~89 ms; prefill mean fell 54.8 → 29.3 ms (1.87×),
  not the earlier projected 3–4×. Prefix reuse is now implemented on Torch.
- Graphs use StaticCache for repeated suffix shapes; prefill stays eager. Suffix
  p50 fell 24.1 → 16.0 ms (8 fields) and 199.8 → 42.1 ms (custom 27 fields).
- Capture cost ~187 / 283 ms respectively, excluded from warmed measurements,
  as are loading and warmup. Keep one model alive to amortize those costs.
- Graphs are off by default; enable `cuda_graph=True` or `PD_CUDA_GRAPH=1`.
  They consume additional memory and fall back to eager on failure. The original
  synthetic 27-field run was refused under memory pressure; the later custom
  run succeeded. **A fresh documentation verification run refused both 8- and
  27-field captures with CUDA reporting zero free bytes.** It ran eager fallback
  successfully but yielded no new replay timings. See
  [`cuda-graph-doc-verification.json`](https://github.com/rorshopping/parallel-decisions/blob/main/benchmarks/cuda-graph-doc-verification.json).
- **Do not multiply** CPU→CUDA, prefix and graph speedups: schemas, lengths,
  precision and summary statistics differ. The earlier CPU comparison includes
  fp32→bf16; process-throughput totals include startup, not just warmed serving.
  Earlier unsynchronized phase timings are not reliable phase breakdowns; the
  newer scripts synchronize CUDA around measurements.
- The browser lab measured ten local classification requests plus DOM updates,
  not ten model-selected actions on real emails. It predates prefix/graph work;
  no combined browser speedup has been measured. Model recommendations remain
  advisory, particularly for consequential actions such as payments.

The merged release suite was verified at **149 passed, 3 skipped** with its source
path pinned explicitly. See the release `AGENTS.md` for the Windows worktree/venv
import trap and test command.

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

