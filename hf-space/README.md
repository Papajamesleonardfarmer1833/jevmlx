---
title: Parallel Constrained Decisions
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
short_description: Typed decisions with confidence from a small local model
models:
- Qwen/Qwen2.5-1.5B-Instruct
---

# Parallel Constrained Decisions

Unofficial demo of a parallel constrained decision engine: pick a preset, press
Run, and get a typed JSON decision with per-field confidence and a latency
breakdown from a small model running locally on the Space. Enable the checkbox to
also run the naive "write the whole JSON document token by token" baseline and
compare validity and speed.

## How it works

- The schema is compiled into a compact catalog of typed fields, and every allowed
  value is mapped to the token that starts it in the model vocabulary.
- One prefill pass embeds the context plus that catalog, then the resulting KV
  cache is broadcast to all fields.
- Every field is decided in a single batched forward pass, so latency stays flat
  as the schema grows (the 255-choice preset demonstrates this).
- Confidence is a softmax restricted to each field's own candidate tokens, so the
  scores are calibrated across the options the field can take.

## Attribution and disclaimer

- Engine code packaged here is copied verbatim from
  [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD)
  (Apache-2.0). See `NOTICE.md` and `LICENSE`.
- This Space is an unofficial demo and is not affiliated with, endorsed by, or
  sponsored by TypeSafe AI.
- Model: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  (Apache-2.0). Set `MODEL_ID` to run a different checkpoint.

## Hardware

Set the Space hardware to **ZeroGPU (zero-a10g)**: the `@spaces.GPU` decorated
entry point needs a GPU, and the app still runs on CPU for local development.
The first request downloads the model weights, so it takes noticeably longer
than later runs.
