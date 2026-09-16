# 02 — TypeSafe AI / Jev: product, docs, manifesto (detail)

> Sources: typesafe.ai, typesafe.ai/blog, typesafe.ai/manifesto, docs.typesafe.ai (fetched 2026-09-16)

## What Jev actually is

An **API-only, early-access model** that takes:

- a **`state`** — unstructured data (a string, a JSON object, program state, a paragraph of context),
- a set of typed **`questions`** — each one a "gut-check" judgment (a highly knowledgeable person could answer in seconds),

and returns **typed answers with probability distributions + confidence** — evaluated **in parallel, in one call**.

No text out. No parsing. No schema violations. That is the whole product thesis: "frontier-intelligence function call: unstructured state in, typed probabilistic decisions out."

## The three primitives (docs.typesafe.ai)

| Question type | Goal | Returns |
|---|---|---|
| **Choice** | Choose an option from a list (up to 255 options) | `choice`, `probabilities`, `confidence` |
| **Score** | Score the state on a rubric | `score`, `probabilities`, `confidence` |
| **Noul** | Is this statement true? | `noul` (0–1) |

Design doctrine from the docs:

- **Atomic questions, composed in code.** "Instead of 'rate this startup pitch', ask separately about market size, technical feasibility, differentiation; combine scores with your own formula."
- Questions are evaluated **independently** → no context-rot as you add questions; adding questions barely changes latency.
- The model is not supposed to do multi-step reasoning in one prompt — the *code* owns the logic, the model owns per-question semantic judgment. "Smart if-statements."

## Positioning vs LLMs (from the launch blog)

| | Existing LLMs | System One + Jev |
|---|---|---|
| Optimized by | RLHF / RLVR | **RLCD (calibrated decisions)** |
| Inputs | messages, sequences | structured program state |
| Outputs | strings (can hallucinate/refuse) | **type-safe structured values** with confidence |
| Sampling | sequential (one token at a time) | **parallel (all outputs in one query)** |
| Latency | 3–329 s | 70–500 ms |
| Cost | $0.20–$10 / MTok input; output ~5x | **$0.042 / MTok input; output free** |
| Failure mode | overconfidence, mode dropping, going off-script | wrong label (but never malformed; confidence exposes uncertainty) |

## Manifesto in 60 seconds ("Build Prod, Not God")

- The bottleneck is not raw intelligence; it's that intelligence is **hard to build on**. Chat models are "horseless carriages" — designed for a human on the other end.
- Goal: **composable AI** — model decisions as dependable as a database query, layerable 5 levels deep, safe because they're typed, inspectable, constrainable.
- Three steps: (1) ship machine-native shape at best intelligence-per-dollar, (2) make it reliable enough to truly automate, (3) offer higher-level abstractions others build on.
- Naming: "System One" from Kahneman (fast, intuitive System 1 vs. deliberate System 2); **Jev** from **Jevons paradox** — cheaper intelligence → more consumption, not less.

## Evals & receipts (their claims, published with caveats)

- **Workflow evals**: assume a "correct compute graph" (workflow in code) and compare each model's decisions to the **reference probabilities from the largest models** (GPT-6 Astra, Fable 5.1 averaged).
- Jev "owns the Pareto frontier" of intelligence-per-dollar: the headline **193.6x faster / 444.6x cheaper** comes from their home page workflow; 4 workflows published at https://evals.typesafe.ai/.
- **Hallucination plot: 0%** for Jev is *by construction* ("schema matching is guaranteed"), not empirical — they say so explicitly.
- **LLM numbers** come from OpenRouter routing → possible selection bias; their adapter constrains LLMs to the same structured format (they argue this is the fairest way to get probabilities out of LLMs, and it's slower/more expensive than normal use).
- Their own disclaimer list is unusually honest (subsidy question, eval-author bias, favorable demo framing) — see `01-...` for the bullet list.

## Early access / ecosystem

- Waitlist at typesafe.ai; playground at console.typesafe.ai (a shareable demo query exists).
- Python adapter for making LLMs emit TypeSafe-compatible structured decisions: `github.com/typesafe-ai/system-one-adapter-python`.
- Docs index: https://docs.typesafe.ai/llms.txt (machine-readable).
- **Nothing open-weights. No paper. No architecture details beyond "new architecture + parallel sampler + RLCD".**

## What you can/can't reproduce locally today

| Piece | Status |
|---|---|
| Typed, parallel, confidence-tagged decisions from a local model | ✅ reproducible (see `04`, `05`) |
| 100% schema validity | ✅ reproducible (programmatic assembly) |
| "Calibrated" confidence | ⚠️ local versions use **softmax over candidate logits** = a *proxy*, not the trained calibration TypeSafe claims |
| RLCD training (whatever it exactly is) | ❌ not public — would require us to design our own RL loop |
| Jev-level model quality | ❌ not public; can only approximate with frontier-ish open models |

## Sources

- https://typesafe.ai/ · https://typesafe.ai/blog/introducing-system-one-models-and-jev · https://typesafe.ai/manifesto
- https://docs.typesafe.ai/ (+ /primitives/choice, /primitives/score, /primitives/noul)
- https://evals.typesafe.ai/ · https://www.theregister.com/ai-and-ml/2026/09/16/typesafe-ai-debuts-model-for-machines-that-plays-doom/5296711
