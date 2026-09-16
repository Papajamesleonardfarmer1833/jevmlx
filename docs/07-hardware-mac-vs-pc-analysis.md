# 07 — Hardware: run it on this Mac, or on the PC (8 GB VRAM / 32 GB RAM)?

**Short answer: run *this* artifact on the MacBook Air (M5 16 GB). It is already working, and its unified memory + MLX fit this workload better than an 8 GB-VRAM GPU. Keep the PC for later, if/when we start *training* or want an independent cross-check.**

## Why — the three things that decide this

### 1. The workload is memory-shaped, not compute-shaped

The engine's expensive operation is **one batched forward pass over a KV cache that was broadcast once per schema field**:

```
KV memory during a parallel call  ≈  (fields) × (context tokens) × (bytes/token/layer) × layers
```

Estimates for the presets we run (≈1,000-token contexts, 28 fields):

| Model | Weights (4-bit) | KV per token (all layers) | KV ×1 (normal) | KV ×28 (broadcast) | Peak total |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | ~1.1 GB | ~28 KB | ~28 MB | ~0.8 GB | **~2 GB** |
| Qwen2.5-7B | ~4.3 GB | ~57 KB | ~57 MB | ~1.6 GB | **~6 GB** |
| Qwen3-8B | ~4.8 GB | ~147 KB¹ | ~150 MB | ~4.1 GB | **~9 GB** |
| Qwen2.5-14B | ~8 GB | ~100 KB | ~100 MB | ~2.8 GB | ~11 GB+ |

¹ Qwen3-8B has 8 KV heads/layer (vs 4 in Qwen2.5-7B) → bigger cache. Estimates, bf16/fp16 KV, 36 layers.

- On the **Mac**, all of that lives in the same 16 GB unified pool — no PCIe copies, and macOS can even swap the cold parts gracefully-ish. 7B fits comfortably; 8B is tight but plausible; 14B is risky.
- On an **8 GB VRAM PC**, the 1.5B case is trivial (≈3 GB in bf16), the **7B case is borderline** (4-bit ≈ 4–5 GB weights + broadcast KV ≈ 1.5–2 GB + CUDA context ≈ 0.5–1 GB → right at the 8 GB cliff, with OOM risk on longer contexts), and 8B is likely out. You'd be forced to offload KV to CPU → latency collapses.

### 2. Software path: MLX-first vs. a port

- The artifact's **primary engine is MLX** (`core/engine_mlx.py`), fusing the broadcast + batched suffix eval into a single Metal call. That's what we measured.
- The **PC path is the port** (`core/engine_torch.py`): correct, but it goes through HuggingFace `DynamicCache.batch_repeat_interleave` + explicit masks, in bf16 — generally slower per FLOP at decode-time than MLX on Apple Silicon, and **requires CUDA + bitsandbytes for 4-bit**. If the PC's GPU is not NVIDIA, the CUDA path doesn't exist at all (CPU fp32 fallback = 6 GB weights + slow).
- The torch engine is designed for **HF Spaces (ZeroGPU A10G)** deployments, not for a 8 GB consumer card.

### 3. Thermals (Air-specific caveat)

The Air is fanless: sustained heavy inference will throttle after minutes. Our benchmark runs are short (seconds), so it doesn't bite. If we ever run long batches (thousands of decisions), expect some derate — measure, don't assume.

## What the PC *is* good for

| Job | Best host | Why |
|---|---|---|
| Running this artifact now | **Mac** | Already works; MLX; unified memory |
| 7B/8B 4-bit inference at 28-field broadcast | **Mac** (probably) | 8 GB VRAM is the binding constraint on the PC |
| Cross-check "does the torch port work the same?" | **PC** | Independent implementation, CUDA    |
| **Training** (LoRA/GRPO on a 1.5B–8B) | **PC** (32 GB RAM) | CUDA ecosystem (peft/bitsandbytes/unsloth); system RAM helps offload; 8 GB VRAM can fine-tune a 1.5B with QLoRA, barely an 8B |
| Training on the Mac | possible but modest | MLX LoRA works (`mlx_lm.lora`), 16 GB limits batch/LoRA rank for ≥7B |

## Concrete recommendation

1. **Do all current inference experiments on the Mac.** We already have the 1.5B and 7B numbers (the 7B ran fine on this Air: 28 fields in 1.5–1.7 s). Add Qwen3-8B as the current ceiling test. Skip 14B on 16 GB.
2. **Don't port the whole thing to the PC yet.** If you want the PC involved, the highest-value version is a **cross-check run of the exact same 3 presets with `core/engine_torch.py`** on the PC (script ready to hand over — see below) so we learn whether CUDA changes the speed/latency story. It's a 15-minute experiment, not a migration.
3. **Revisit the PC when we get ambitious**: real RLCD-style training (even a modest GRPO/LoRA run) or the "leaked" 9B-class VL model — that's where 32 GB system RAM is a real advantage and the Mac starts to sweat.

### Ready-to-paste PC cross-check (one-time)

```bash
# on the PC, in a fresh clone of this research folder
python3 -m venv .venv && source .venv/bin/activate   # or Windows equiv.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.40 accelerate>=0.28 numpy>=1.26
cd source/Qwen-2.5-1B-RLCD
BACKEND=torch PYTHONPATH=. python -m core.benchmark    # auto-falls back to torch on non-mac
```

Expected: same numbers ±30% for the 1.5B, device shows `cuda`; the interesting question is whether the 28-field broadcast OOMs at 7B — which we predict it will, near the context limits.

## Disk / environment notes (this Mac)

- 115 GB free before the runs; 1.5B-4bit ≈ 1 GB, 7B-4bit ≈ 4.3 GB, Qwen3-8B-4bit ≈ 4.8 GB.
- Models cache in `~/.cache/huggingface/hub` — prune with `huggingface-cli delete-cache` if needed.
