# Jev-style decisions on a laptop

**Unofficial research repo. Not affiliated with TypeSafe AI.** This is a hands-on study of the *parallel constrained decoding* idea behind [Jev](https://typesafe.ai/) — the "System One" model launched by Diogo Almeida's TypeSafe AI in Sep 2026 — reproduced on a stock, untrained small model, on an Apple Silicon laptop.

The trick: instead of generating JSON token-by-token, define your schema as a set of typed fields, prefill the context **once**, broadcast the KV cache across one batch row per field, and evaluate **all fields in a single forward pass**. JSON is assembled programmatically, so it can never be malformed. The interesting claim: **a 1.5B model that cannot reliably write 28-field JSON can still make 28 schema-valid decisions in ~0.4 s.**

```
[context + schema] ──► prefill (once) ──► KV cache
                                            │ broadcast ×N fields
      ┌───────────┬───────────┬─────────────┴───────────┐
    field 1     field 2     ...                      field N
      └───────────┴──── one batched forward pass ────────┘
                          │
              slice logits → softmax → pick value + probability
```

Engine: [`harshatheg/Qwen-2.5-1B-RLCD`](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) (Apache-2.0, cloned at setup, **not** vendored here). We wrote our own runner, benchmarked three model sizes on a 16 GB M5 MacBook Air, and documented what's real vs. marketing.

## Quickstart (Apple Silicon Mac)

```bash
git clone https://github.com/rorshopping/jev-on-a-laptop
cd jev-on-a-laptop
./setup.sh          # creates .venv, installs mlx-lm, clones the upstream engine
./run_benchmark.sh  # default: Qwen2.5-1.5B-Instruct-4bit
```

Non-Mac? The upstream engine also has a PyTorch/CUDA path (and a Hugging Face Space version of this demo: [link pending]). See `docs/07-hardware-mac-vs-pc-analysis.md` for the memory math behind our Mac-vs-PC recommendation.

Try a single decision instead:

```bash
.venv/bin/python tools/demo.py                          # fintech fraud preset
.venv/bin/python tools/demo.py support_triage --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Sample output:

```
field                  value               conf
---------------------  ------------------  -----
is_fraudulent          True                0.999
risk_tier              CRITICAL            0.997
recommended_action     FREEZE_ACCOUNT      0.982
...
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

## Is a bigger model worth it?

Speed: 1.5B → 7B triples latency (0.41 → 1.52 s); 8B adds ~33% more (2.03 s) and showed no clear behavioral win. Prefill dominates at 28 fields. See `quality-eval/` for the labeled accuracy comparison that answers this properly (built into this repo).

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

## Repo layout

```
setup.sh, run_benchmark.sh   one-command setup + benchmark (Mac)
tools/bench_model.py         benchmark any mlx-lm model, saves results/*.json
tools/demo.py                single decision call, pretty-printed
quality-eval/                labeled accuracy + calibration comparison (1.5B/7B/8B)
hf-space/                    self-contained Gradio Space demo (torch/ZeroGPU)
x-posts/                     copy-paste-ready posts + optional Playwright helper
docs/                        full research notes (start at 04 → 05 → 07 → 08)
results/                     raw benchmark outputs
```

## Credits & license

- Engine: [harshatheg/Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) — Apache-2.0, cloned by `setup.sh`, not redistributed here.
- Jev / RLCD (Reinforcement Learning for Calibrated Decisions): [TypeSafe AI](https://typesafe.ai/) — we have no affiliation and no access to their model. "RLCD" here refers to the community parallel-decoding recreation.
- Our code and docs: MIT (see `LICENSE`).
