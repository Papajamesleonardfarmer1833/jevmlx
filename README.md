# openjev

[![CI](https://github.com/bnsd55/openjev/actions/workflows/ci.yml/badge.svg)](https://github.com/bnsd55/openjev/actions/workflows/ci.yml) [![Build](https://github.com/bnsd55/openjev/actions/workflows/build.yml/badge.svg)](https://github.com/bnsd55/openjev/actions/workflows/build.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

Jev-style parallel constrained decisions for any MLX model on Apple Silicon. Typed, schema-valid JSON in one forward pass.

openjev makes any local instruct model (Qwen, Llama, Mistral, Gemma via mlx-lm) answer a typed schema in a single batched forward pass: prefill the context once, broadcast the KV cache across one row per field, pick each value from its allowed choices. JSON is assembled, never generated, so it is always valid. A 1.5B model decides 28 fields in ~0.4 s on a MacBook Air.

Unofficial. Not affiliated with TypeSafe AI or Jev. Based on the research in rorshopping/jev-on-a-laptop, see NOTICE.

```
[context + schema] ──► prefill (once) ──► KV cache
                                            │ broadcast ×N fields
      ┌───────────┬───────────┬─────────────┴───────────┐
    field 1     field 2     ...                      field N
      └───────────┴──── one batched forward pass ────────┘
                          │
              slice logits → softmax → pick value + probability
```

## Quickstart (Apple Silicon Mac)

```bash
git clone https://github.com/bnsd55/openjev && cd openjev
./setup.sh
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

## Python API

Pydantic models define the schema; `openjev.decide` returns a typed, validated decision with per-field confidences. Fields may be `bool`, `Literal[...]`, or `enum.Enum` with string values. Pydantic validates the result, so a bad value raises instead of leaking through.

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

## Calibration

Raw confidences are softmax(scores) at T=1 and run overconfident. One scalar temperature, fitted by minimizing NLL on labeled data, fixes most of it — no other tuning. Build a labeled JSONL (`{"schema": ..., "context": ..., "labels": {field: value}}` per line) from the repo's labeled cases with `benchmarks/to_jsonl.py`, then:

```bash
.venv/bin/python benchmarks/to_jsonl.py > cases.jsonl
openjev calibrate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --data cases.jsonl
```

Pass the fitted value to decisions with `openjev decide --temperature T`.

Example output, run on the repo's 24 labeled cases (`benchmarks/cases.json`, 72 field decisions, Qwen2.5-1.5B-Instruct-4bit):

```
n samples      : 72
fitted T       : 1.7178
ECE before     : 0.0870  (T=1.0)
ECE after      : 0.0773  (T=1.7178)
accuracy       : 0.5972
```

## Background

openjev applies parallel constrained decoding: prefill the context once, broadcast the KV cache across one row per schema field, and pick every value from its allowed choices in a single batched forward pass — the JSON object is assembled, never generated.

The original research — the full benchmark study, hardware notes, and the head-to-head with Jev itself — lives in the upstream repository [rorshopping/jev-on-a-laptop](https://github.com/rorshopping/jev-on-a-laptop).

Our own cross-model measurements are in [Model compatibility](#model-compatibility) above.

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

Where this is going next: [ROADMAP.md](ROADMAP.md) (calibration, evaluation expansion, packaging, integrations).

## HTTP server

`openjev serve --model mlx-community/Qwen2.5-1.5B-Instruct-4bit` loads the model once and serves one decision per request on `POST /decide` (stdlib `http.server`, serial — one Metal GPU).

```bash
curl -s localhost:8000/decide -H 'Content-Type: application/json' \
  -d '{"schema": {"action": {"type": "enum", "description": "The action to take", "choices": ["APPROVE", "BLOCK_TRANSACTION"]}}, "context": "payment from a verified customer, all checks passed", "temperature": 1.0}'
```

`GET /health` returns `{"ok": true, "model": M}`. Bad JSON, missing keys, and invalid schemas return 400; anything else returns 500 with the exception's first line.

## Repo layout

```
openjev/                     engine, schema, typed API, calibration, CLI
openjev/presets/             example decision schemas
benchmarks/                  compat matrix, naive-vs-parallel bench, labeled cases
setup.sh                     one-command setup (Apple Silicon Mac)
tests/                       pytest suite (fast tests + model-marked slow tests)
.github/                     CI/build workflows, issue and PR templates
```

## Credits & license

- Jev / RLCD (Reinforcement Learning for Calibrated Decisions): [TypeSafe AI](https://typesafe.ai/) — we have no affiliation and no access to their model. "RLCD" here refers to the community parallel-decoding recreation.
- Origins and third-party credits: see `NOTICE`.
- Our code and docs: MIT (see `LICENSE`).
