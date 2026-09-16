# Head-to-head eval on TypeSafe's own questions

This folder reproduces a like-for-like comparison against the published evals at
**https://evals.typesafe.ai/** (TypeSafe AI, Jev / "System One" models).

## What it does

1. `extract_typesafe_mini.py` — downloads the public viewer data for the
   **Security Incidents** workflow (question catalog, five example cases, every
   model's answers, and the Astra+Fable consensus reference) and normalizes it
   into `typesafe/mini_eval.json`. **The raw case documents are not redistributed
   in this repo**; the script fetches them from TypeSafe's public site.
2. `make_subagent_prompts.py` — writes one self-contained prompt per case
   (`prompts/case_*.txt`) containing only the input state + question catalog +
   output contract. Nothing about the reference is included.
3. Answer the prompts with any model (we used subagents on the session model,
   DeepSeek v4.1 Flash via opencode-go at `variant: max`, one call per case,
   ~$0.03 total).
4. `record_subagent_answers.py` — normalizes recorded answers
   (`subagent_answers.json`) into `results/deepseek-v4.1-flash.json`.
5. `eval_local.py` — runs the same questions through the local parallel-decisions
   engine with an MLX model (default `mlx-community/Qwen3-8B-4bit`).
6. `score_eval.py` — agreement against the consensus reference for every model,
   including a strict common-subset table for like-for-like comparison.

## Results (2026-09-16, 5 cases / 48 reference pairs)

Strict common subset — the 26 (case, question) pairs answered by every model:

```
model            agree   n     acc
--------------------------------------
sol                 23  26   88.5%
opus                21  26   80.8%
typesafe (Jev)      20  26   76.9%
local-qwen3-8b      20  26   76.9%
```

The local 8B (Qwen3-8B-4bit on an M5 MacBook Air) ties Jev on Jev's own curated
cases against the frontier consensus, at ~$0 marginal cost and ~0.4–0.6 s/case.

**Caveats:** 5 curated *disagreement* cases (chosen because models differ), so
absolute accuracy is not comparable to TypeSafe's published per-workflow numbers
(61.7–76.0%). The reference is a consensus of two frontier models — agreement
with it is not correctness. n=26; treat as an anecdote, not an estimate.

Full write-up: see `docs/11-typesafe-eval-head-to-head.md` in this repository.
