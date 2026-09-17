# openjev

[![CI](https://github.com/bnsd55/openjev/actions/workflows/ci.yml/badge.svg)](https://github.com/bnsd55/openjev/actions/workflows/ci.yml) [![Build](https://github.com/bnsd55/openjev/actions/workflows/build.yml/badge.svg)](https://github.com/bnsd55/openjev/actions/workflows/build.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Jev-style parallel constrained decisions for any MLX model on Apple Silicon. Typed, schema-valid JSON in one forward pass.

**Inspired by and built on [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop)**, the research repo that reproduced the technique on a laptop. openjev turns that study into an installable library. Unofficial, not affiliated with TypeSafe AI or Jev.

openjev makes any local instruct model (Qwen, Llama, Mistral, Gemma via mlx-lm) answer a typed schema in a single batched forward pass: prefill the context once, broadcast the KV cache across one row per field, pick each value from its allowed choices. JSON is assembled, never generated, so it is always valid. A 1.5B model decides 28 fields in under a second on an Apple Silicon laptop (see [Model compatibility](#model-compatibility)).

```
[context + schema] ──► prefill (once) ──► KV cache
                                            │ broadcast ×N fields
      ┌───────────┬───────────┬─────────────┴───────────┐
    field 1     field 2     ...                      field N
      └───────────┴──── one batched forward pass ────────┘
                          │
              slice logits → softmax → pick value + probability
```

## Install

| Method | Command | Notes |
|---|---|---|
| pip (git) | `uv pip install git+https://github.com/bnsd55/openjev` | Installs the `openjev` package and CLI |
| From source | `git clone https://github.com/bnsd55/openjev && cd openjev && ./setup.sh` | Creates `.venv`, editable install with dev extras |
| Requirements | — | Apple Silicon Mac (macOS, arm64), Python 3.12; engine fails fast elsewhere |

## Quickstart (Apple Silicon Mac)

```bash
.venv/bin/openjev decide --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --preset fintech_fraud
```

Sample output:

```
Preset : FinTech Fraud & Autonomous AML Compliance (28 Fields)
Model  : mlx-community/Qwen2.5-1.5B-Instruct-4bit
Latency: 970.9 ms (prefill 416.6 + batched pass 530.4)

field                           value                   conf   type
------------------------------  ----------------------  -----  -----
is_fraudulent                   True                    0.825  boolean
risk_tier                       HIGH                    0.752  enum
recommended_action              BLOCK_TRANSACTION       0.702  enum
...
```

## Features

**01. One forward pass for every field.** The context is prefilled once, then the KV cache is broadcast so all schema fields are scored in a single batched pass. Latency grows with the longest suffix, not with field count.

**02. Always-valid JSON.** The JSON object is assembled programmatically from per-field decisions — it is never generated token by token, so it cannot be malformed. Every value comes from the field's allowed choices.

**03. Honest confidence.** Each field reports softmax over the per-choice scores at the decision position — no clamps, no rounding tricks. Raw probabilities run overconfident; [Calibration](#calibration) fits one temperature to fix that.

**04. Typed Python API.** Pass a Pydantic model, get a validated instance back with per-field confidences:

```python
from typing import Literal
from pydantic import BaseModel, Field
import openjev


class Fraud(BaseModel):
    is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")


context = "Wire transfer to a new IBAN, requested from a Tor exit node on an unrecognized device"

d = openjev.decide(Fraud, context, model="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
d.value  # Fraud(is_fraudulent=True, risk_tier="CRITICAL")
d.confidence  # {"is_fraudulent": 0.99, "risk_tier": 0.97}
d.latency_ms
```

**05. HTTP server.** `openjev serve --model M` loads the model once and serves one decision per request on `POST /decide` (stdlib `http.server`, serial — one Metal GPU):

```bash
curl -s localhost:8000/decide -H 'Content-Type: application/json' \
  -d '{"schema": {"action": {"type": "enum", "description": "The action to take", "choices": ["APPROVE", "BLOCK_TRANSACTION"]}}, "context": "payment from a verified customer, all checks passed", "temperature": 1.0}'
```

`GET /health` returns `{"ok": true, "model": M}`; bad input returns 400.

**06. Temperature calibration.** `openjev calibrate` fits one scalar temperature on labeled JSONL data by minimizing NLL, then reports binned ECE before and after:

```
n samples      : 72
fitted T       : 1.7178
ECE before     : 0.0870  (T=1.0)
ECE after      : 0.0773  (T=1.7178)
accuracy       : 0.5972
```

**07. Works with any mlx-lm instruct model.** Prompts are built with the tokenizer's own chat template — no hand-rolled role tags, no system-role assumptions. Cross-model measurements: [Model compatibility](#model-compatibility).

## How it works

1. **Prefill once.** The context plus a compact schema catalog goes through the model a single time ([engine.py](openjev/engine.py)).
2. **One row per field.** Each field's choice suffixes are teacher-forced as rows against the broadcast KV cache; choices that share a first token get their own rows.
3. **Memory guard.** Row batches are auto-chunked over the same prefill cache so peak Metal memory stays bounded.
4. **Batched pass.** All rows are evaluated in one forward pass.
5. **Scoring.** Each field's logits are sliced at its decision position, softmaxed over its choices, and the JSON object is assembled from the winners.

## Model compatibility

Every preset field is decided in one batched pass — measured on a MacBook Pro M2 Pro, 34 GB, macOS (warm runs, `benchmarks/compat.py`). "Warm latency" is the average of the second run on both presets; peak memory is Metal's process high-water mark after both presets.

| Model | loads | presets valid | warm latency (ms, avg of 2 presets) | prompt tokens | peak GPU mem (GB) |
|---|---|---|---|---|---|
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | y | ok, ok | 862 | 469 | 2.26 |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | y | ok, ok | 3785 | 469 | 6.30 |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | y | ok, ok | 1701 | 422 | 5.03 |
| `mlx-community/gemma-2-2b-it-4bit` | y | ok, ok | 1407 | 474 | 4.83 |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | y | ok, ok | 5519 | 557 | 10.32 |
| `mlx-community/Phi-3.5-mini-instruct-4bit` | y | ok, ok | 7562 | 569 | 13.73 |

## Benchmarks

Three scripts in [benchmarks/](benchmarks/), each run against a local mlx-lm model id:

- `compat.py` — the cross-model compatibility table above.
- `naive_vs_parallel.py` — autoregressive JSON baseline vs the parallel engine, per preset.
- `cases.json` — 24 labeled cases (72 field decisions) behind the calibration numbers; `to_jsonl.py` converts them for `openjev calibrate`.

## Roadmap

Where this is going next: [ROADMAP.md](ROADMAP.md). Latency profiling, an evaluation loop, `openjev serve` completion, more field types on request, then PyPI.

## Contributing

Branch off `main`, run `ruff check --fix . && ruff format .` and `pytest -m "not slow"` before pushing. Details: [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits & license

openjev is MIT-licensed (see [LICENSE](LICENSE)); third-party credits are listed in [NOTICE](NOTICE).

- [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop) (MIT) — the research origin and inspiration; the study that reproduced the technique on a laptop.
- [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) (Apache-2.0) — the original parallel constrained decoding engine whose approach openjev reimplements.
- [TypeSafe AI's Jev](https://typesafe.ai/) — the product whose published technique this project reimplements; no affiliation, no access to their model.
