# Deploying the interactive Gradio Space

**Status: ready but blocked on Hugging Face PRO.**

The self-contained app in this folder runs the live engine (PyTorch/ZeroGPU or CPU) in the
browser. Uploading it was attempted on 2026-09-16 and Hugging Face returned:

> 402 Payment Required — "Static Spaces are free for everyone, but hosting Gradio and Docker
> Spaces on free cpu-basic requires a PRO subscription."

So: **static Spaces are free (the live showcase uses one); Gradio Spaces need PRO.**

## If you have PRO

```bash
cd hf-space
pip install huggingface_hub          # if not already
hf auth login                        # or: HF_TOKEN=hf_... python push_to_hub.py
python push_to_hub.py
```

The script creates `rorshopping/parallel-constrained-decisions` (or `<your-user>/...`),
uploads everything, and best-effort requests `zero-a10g` hardware. If hardware assignment
fails, set it manually in the Space settings: **Settings → Hardware → ZeroGPU**.

Resulting URL: `https://huggingface.co/spaces/<username>/parallel-constrained-decisions`.

## Locally, without any Hugging Face account

```bash
cd hf-space
python -m venv .venv-dev
. .venv-dev/bin/activate
pip install -r requirements.txt
BACKEND=torch python app.py     # opens http://localhost:7860
```

It will download Qwen2.5-1.5B-Instruct (~3 GB in fp32) and run on CPU — slow (tens of seconds
per call) but functional. On Apple Silicon you can instead run the much faster MLX path from the
parent repo (`./setup.sh` then `tools/demo.py`), or the upstream web UI:

```bash
cd ../vendor/Qwen-2.5-1B-RLCD
PYTHONPATH=. ../../.venv/bin/python -m uvicorn server.app:app --port 8000
```

## What the user sees when it is live

- pick a preset, edit the context, press Run
- per-field table: value, confidence, runners-up
- assembled JSON + latency split (prefill / batched pass / total)
- optional checkbox to also run the naive JSON baseline for the side-by-side contrast
