# 04 — The HF artifact: what `harshatheg/Qwen-2.5-1B-RLCD` actually is

> Verified 2026-09-16 by cloning the repo and inspecting every file, plus the HF API records for the model and its Space.

## TL;DR — read this first

**It is not a trained model.** Despite the name, the "model" repo contains **zero weights**:

- HF API: `"usedStorage": 0`, and the file list has **no `.safetensors`, no `model.npz`, no adapters**.
- The git repo we cloned is **~112 KB of pure code** (16 Python/JSON/JS files + docs).
- At runtime the engine downloads and loads a **stock** `mlx-community/Qwen2.5-1.5B-Instruct-4bit` from the MLX community.
- There is **no training code**, no RL, no RLCD anywhere in the repo. The only "RLCD" occurrences are **aliases** (`run_rlcd_generation = run_parallel_generation`, `to_rlcd_schema_str = to_parallel_schema_str`).

So: it's a **parallel constrained decoding engine** + benchmark + demo UI, built to *approximate the output shape* of TypeSafe's Jev using an off-the-shelf Qwen model. That's a legitimate and interesting artifact — but it is **not** a reproduction of TypeSafe's RLCD training, and it is **not** a fine-tune you can compare with Jev.

## Repo inventory

| File | Size | Purpose |
|---|---|---|
| `README.md` / `MODEL_CARD.md` | 12.1 KB / 5.5 KB | Same content — the marketing doc (benchmarks, SDK docs) |
| `core/engine.py` | 1.1 KB | Backend router: **MLX on macOS**, **PyTorch/CUDA elsewhere** |
| `core/engine_mlx.py` | 16.2 KB | The real engine (Apple Silicon) |
| `core/engine_torch.py` | 11.0 KB | Torch/CUDA/CPU port (used for the Docker/HF Space deployment) |
| `core/schema.py` | 9.1 KB | Schema types (boolean/enum), candidate-token compilation |
| `core/prompt_builder.py` | 2.4 KB | Prompts for the naive JSON baseline |
| `core/benchmark.py` | 3.1 KB | CLI benchmark runner |
| `presets/*.json` | 4 files | Fintech (28 fields), Support Triage (28), Code Security (28), High Cardinality (4 fields, 255 choices) |
| `server/app.py`, `server/main.py` | 5.2 KB | FastAPI server (`/api/run-parallel`, `/api/stream-naive`) |
| `web/*` | ~22 KB | Side-by-side visualizer |
| `Dockerfile`, `run.sh`, `requirements*.txt` | — | HF Spaces (ZeroGPU) + local run |

## Metadata vs. reality

| Claim/metadata | Reality |
|---|---|
| `pipeline_tag: text-generation` | It's an inference harness, not a generative model |
| `library_name: mlx` | True — but the **weights are mlx-community's**, not the author's |
| `base_model: Qwen/Qwen2.5-1.5B-Instruct` | The author tags a base model, implying a finetune — **no finetuned weights exist here** |
| Model ID says "1B", text says "1.5B" | Sloppy naming; the engine loads the **1.5B** 4-bit |
| Tags: `calibrated`/`classification`/`structured-generation` | Structurally true; calibration is a softmax proxy, not trained calibration (see `05`) |
| "5.6x–7.0x speedup, 100% schema validity" | Reproduced in spirit on our M5: **4.4x–7.9x** vs. their own naive baseline (see `06`) |
| HF Space `drinkmoonshine/parallel-constrained-decoding` | Exists, **paused** at fetch; ZeroGPU A10G; same file tree |

## The honest framing for the KB

```
TypeSafe claims                      This artifact
────────────────────────────         ─────────────────────────────────────
RLCD-trained model (Jev)             Stock Qwen2.5-1.5B-Instruct-4bit, 0 training steps
New architecture + parallel sampler  Parallel-ish sampler: broadcast KV + batched logits
Calibrated probabilities (trained)   Softmax over candidate token logits (proxy)
Type-only outputs, no strings        Yes — programmatic JSON assembly, same property
"Can't hallucinate" type errors      Holds for keys/enums; collision path has fallbacks
40–200x faster, $0.042/MTok          N/A (local); 4.4–7.9x vs. its own naive baseline
```

## Provenance / who made it

- HF user: **harshatheg** (the "model" author); Space by **drinkmoonshine** (probably the same person or a collaborator — unverified). There is no public X thread, GitHub repo, or blog post we could find that documents a training run.
- Published **2026-09-16 01:35 UTC** — i.e. **within hours of the TypeSafe launch** (which was Sep 15, ~19:00 UTC). It is a same-day community reaction, not the original research.
- 31 likes / trending, 0 tracked downloads at fetch time.
- The README's `git clone https://github.com/your-org/parallel-constrained-decoding.git` is a **placeholder URL** — there is no public GitHub origin. The HF repo is the only source.
- The in-repo alias hints the author originally called this "RLCD" before renaming to "Parallel Constrained Decoding".

## Implication for our project

Since it is just a harness on top of stock models:

1. **We can swap models freely** (any `mlx-lm` model with a ChatML-ish template) — done for 7B/Qwen3 in `08`.
2. **We can improve it ourselves** (real calibration, better collision handling, more field types) without touching weights.
3. **The interesting open question is not "can it run"** (it does, today, on your Mac) **but** "how good are the decisions and how honest are the probabilities" — that's the evaluation plan in `09`.
