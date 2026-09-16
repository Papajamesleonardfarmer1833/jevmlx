---
title: Parallel Constrained Decisions
emoji: ⚡
colorFrom: green
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Typed decisions with confidence in one batched pass
models:
- mlx-community/Qwen2.5-1.5B-Instruct-4bit
---

# Typed decisions, one pass, no parsing

**Unofficial demo. Not affiliated with TypeSafe AI.**

Static showcase of the *parallel constrained decoding* idea behind Jev (TypeSafe AI, Sep 2026):
instead of generating JSON token-by-token, a schema is defined as typed fields, the context is
prefilled **once**, the KV cache is broadcast across one batch row per field, and **all fields are
evaluated in a single batched forward pass**. The JSON is assembled programmatically, so it can
never be malformed.

What's on this page (recorded output from a stock, untrained **Qwen2.5-1.5B-Instruct-4bit** on an
M5 MacBook Air, 16 GB):- 4 realistic presets (fraud routing, support triage, code security, 255-choice tariff)
- per-field decisions with confidence and runners-up
- assembled JSON next to the naive baseline (the same model writing the whole JSON itself)
- latency split (prefill + batched pass) and naive timings

Highlights: 28-field decisions in ~0.4 s, 7–8x faster than naive JSON generation — and the naive
baseline produced malformed/incomplete JSON on 3 of the 4 presets while the constrained path was
always schema-valid. Model size does not fix JSON reliability; decoding structure does.

**Honest caveats:** confidence here is a softmax over candidate token logits — a proxy for
calibration, not trained calibration (that's what TypeSafe's RLCD training is for). Fields whose
choices share a first token hit a slower fallback path.

Live interactive version (Gradio, ZeroGPU) and the full write-up, benchmarks for 1.5B/7B/8B, and
raw logs: https://github.com/rorshopping/jev-on-a-laptop

Engine credit: [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) (Apache-2.0),
cloned at runtime — not redistributed here.
