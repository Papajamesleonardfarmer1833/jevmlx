# openjev

Jev-style parallel constrained decisions for any MLX model on Apple Silicon. Typed, schema-valid JSON in one forward pass.

openjev makes any local instruct model (Qwen, Llama, Mistral, Gemma via mlx-lm) answer a typed schema in a single batched forward pass: prefill the context once, broadcast the KV cache across one row per field, pick each value from its allowed choices. JSON is assembled, never generated, so it is always valid. A 1.5B model decides 28 fields in ~0.4 s on a MacBook Air.

Unofficial. Not affiliated with TypeSafe AI or Jev. Fork of rorshopping/jev-on-a-laptop, see NOTICE.

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
d.value            # Fraud(is_fraudulent=True, risk_tier="CRITICAL")
d.confidence       # {"is_fraudulent": 0.99, "risk_tier": 0.97}
d.latency_ms
```

## Calibration

Raw confidences are softmax(scores) at T=1 and run overconfident. One scalar temperature, fitted by minimizing NLL on labeled data, fixes most of it — no other tuning. Build a labeled JSONL (`{"schema": ..., "context": ..., "labels": {field: value}}` per line) from the repo's labeled cases with `tools/quality_eval_to_jsonl.py`, then:

```bash
.venv/bin/python tools/quality_eval_to_jsonl.py > cases.jsonl
openjev calibrate --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --data cases.jsonl
```

Pass the fitted value to decisions with `openjev decide --temperature T`.

Measured on the repo's 24 labeled quality-eval cases (72 field decisions, Qwen2.5-1.5B-Instruct-4bit, via `tools/quality_eval_to_jsonl.py`):

```
n samples      : 72
fitted T       : 1.7178
ECE before     : 0.0870  (T=1.0)
ECE after      : 0.0773  (T=1.7178)
accuracy       : 0.5972
```

## Measured results (M5 MacBook Air, 16 GB)

28-field fraud preset. "Naive" = the same model writing the whole JSON object token-by-token.

| Model (4-bit) | Naive JSON | Parallel decisions | Speedup | Naive schema-valid? | Parallel schema-valid? |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 3.3 s | **0.41 s** | 7.9x | ❌ | ✅ |
| Qwen2.5-7B | 11.9 s | **1.52 s** | 7.9x | ❌ | ✅ |
| Qwen3-8B | 14.3 s | **2.03 s** | 7.0x | ❌ | ✅ |

Also measured: 4-field / 255-choice tariff preset (1.5B: 0.15 s, 5.9x) and support-triage (1.5B: 0.76 s, 4.4x — collision handling costs extra). Full logs and JSON: `results/`.

**Punchline: none of the three models could reliably emit 28-field JSON unconstrained — all three are always schema-valid through the constrained path.** Model size does not fix JSON reliability; the decoding structure does.

> Latency numbers in the older sections come from the upstream M5 MacBook Air study; the table below was measured on the current development machine.

## Model compatibility

Every preset field is decided in one batched pass — measured on a MacBook Pro M2 Pro, 34 GB, macOS (warm runs, `tools/compat.py`). "Warm latency" is the average of the second run on both presets; peak memory is Metal's process high-water mark after both presets.

| Model | loads | presets valid | warm latency (ms, avg of 2 presets) | prompt tokens | peak GPU mem (GB) |
|---|---|---|---|---|---|
| `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | y | ok, ok | 862 | 469 | 2.26 |
| `mlx-community/Qwen2.5-7B-Instruct-4bit` | y | ok, ok | 3785 | 469 | 6.30 |
| `mlx-community/Llama-3.2-3B-Instruct-4bit` | y | ok, ok | 1701 | 422 | 5.03 |
| `mlx-community/gemma-2-2b-it-4bit` | y | ok, ok | 1407 | 474 | 4.83 |
| `mlx-community/Mistral-7B-Instruct-v0.3-4bit` | y | ok, ok | 5519 | 557 | 10.32 |
| `mlx-community/Phi-3.5-mini-instruct-4bit` | y | ok, ok | 7562 | 569 | 13.73 |

## Is a bigger model worth it? (measured)

24 labeled cases × 3 fields = 72 decisions per model, run through the same engine (`quality-eval/`):

| Model | Primary field acc | All fields exact | Latency/case |
|---|---|---|---|
| 1.5B | 58% (= majority-class baseline 54%) | 50% | 147 ms |
| **7B** | **96%** | 72% | 611 ms |
| **8B** | 92% | **85%** | 646 ms |

**Verdict:** yes, 1.5B→7B is a clear win (+37 pts primary accuracy); 1.5B is only suitable for demos/UI work. 7B is the best when the single primary decision is what you act on; 8B is best when *all* fields must be jointly correct, at nearly the same speed. Also measured: **confidence does not reliably flag errors** — the 7B was >0.90 confident on 13 of its 20 wrong fields. Full report: `quality-eval/SUMMARY.md`.

## What's real vs. what's marketing (in our measurements)

| Claim | Verdict here |
|---|---|
| Typed outputs, no string generation | ✅ by construction |
| 100% schema validity | ✅ keys/enums guaranteed; values can still be *wrong* |
| One pass for all fields | ✅ common case; fields whose choices share a first token hit a slow fallback |
| "Calibrated" confidence | ❌ raw softmax over candidate logits — a proxy, not trained calibration |
| 40–200x faster | N/A for local; vs. its own naive baseline we measured **3.4–7.9x** |
| Runs on a laptop | ✅ this exact repo |

More analysis in [`docs/`](docs/) — the research notes, hardware fit tables, and the failure-mode list.

## How does this compare to Jev itself?

TypeSafe's own workflow evals (https://evals.typesafe.ai/) put Jev at **67.8% mean accuracy** (61.7–76.0% per workflow) against a **frontier consensus** (average of GPT-6 Astra and Fable 5.1 answering every question), at $0.0004 and 0.4 s per case. Our 8B's 84.7%/91.7% is **not comparable** (rule-constructed labels, synthetic cases, n=24) — it would imply beating Opus 5 (73.1%) and Sol (74.1%) on their eval, which is implausible. Comparable findings: same-order latency (0.4 s vs 0.65 s) and a ~100–1000x local cost advantage, both type-safe by construction. Full analysis in `docs/10-jev-published-accuracy-vs-our-8b.md`.

## 5. Head-to-head on TypeSafe's own questions (full public eval)

We rebuilt **all public example cases of all four TypeSafe workflows** — 20 cases, **373 reference question-pairs** — and scored everyone against TypeSafe's own reference (consensus of GPT-6 Astra + Fable 5.1). Full detail: `evals/RESULTS.md` and `docs/12-full-head-to-head.md`.

Strict like-for-like — the 343 pairs answered by every model:

| Model | Agreement | Pairs |
|---|---|---|
| Opus (published) | 89.8% | 308/343 |
| DeepSeek v4.1 Flash (max) | 89.5% | 307/343 |
| Sol (published) | 89.2% | 306/343 |
| **Jev / TypeSafe (published)** | **86.6%** | **297/343** |
| **local Qwen2.5-7B (free, on an M5 Air)** | **73.8%** | **253/343** |
| local Qwen3-8B (3 of 4 workflows) | 71.2% | 114/160 |

**The honest result: the free local model is ~13 points behind Jev on Jev's own benchmark.** An earlier 5-case run (`docs/11-...`) showed a tie, but that was an artifact of the tiny curated sample — on the full public set the gap is real and stable across all four workflows. The local model's advantages are cost (~$0), privacy, and offline operation, not accuracy parity. The frontier cluster sits at 86–90%; Jev at 86.6% is genuinely in that cluster at 1/1000th the price.

Where this is going next: [ROADMAP.md](ROADMAP.md) (calibration, evaluation expansion, packaging, integrations).

## Repo layout

```
openjev/                     engine, schema, typed API, calibration, CLI + example presets
setup.sh, run_benchmark.sh   one-command setup + benchmark (Mac)
tools/                       benchmark scripts
quality-eval/                labeled accuracy + calibration comparison (1.5B/7B/8B)
evals/                       head-to-head on TypeSafe's published security-incident questions
results/                     raw benchmark outputs
docs/                        full research notes (start at 04 → 05 → 07 → 08)
```

## Credits & license

- Jev / RLCD (Reinforcement Learning for Calibrated Decisions): [TypeSafe AI](https://typesafe.ai/) — we have no affiliation and no access to their model. "RLCD" here refers to the community parallel-decoding recreation.
- Origins and third-party credits: see `NOTICE`.
- Our code and docs: MIT (see `LICENSE`).
