# 11 — Head-to-head on TypeSafe's own security-incident questions (5 curated cases)

> Superseded as the headline result by `12-full-head-to-head.md`, which runs **all four workflows,
> 20 cases, 373 question-pairs**. The 5-case read here (local 8B tying Jev at 20/26) turned out to be
> an artifact of the tiny curated sample: on the full set the local model is ~13 points behind Jev.
> Kept for provenance of the method and as the worked example.

> Run 2026-09-16. Method, results, and the caveats that matter.
> Data: `evals/typesafe/mini_eval.json` (extracted from the public viewer at evals.typesafe.ai)

## What we did

TypeSafe's public eval viewer ships, for its **Security Incidents** workflow:

- the **question catalog** — 14 typed questions (11 Noul probabilities, 1 Score 0–3, 2 Choice);
- the **state** for the five example cases it curates (alert + joined context records: AD, CrowdStrike, Okta, ServiceNow, Tanium, Workday, Lenel…);
- **every model's answers** (Opus, Sol, Jev/TypeSafe) on those cases;
- the **reference answers**: per-question probability distributions from the consensus of **GPT-6 Astra (high thinking) and Claude Fable 5.1 (high)**, averaged.

We rebuilt the same five cases through three additional answerers and scored everyone identically — **agreement between the model's answer and the consensus argmax**, per (case, question) pair:

| Answerer | How it answered |
|---|---|
| **opus / sol / typesafe (Jev)** | published answers, taken from the viewer data |
| **deepseek-v4.1-flash** | five subagent calls on the session model (`opencode-go`, variant `max`), one per case, all questions in one prompt |
| **local-qwen3-8b** | the local MLX parallel-decisions engine (`evals/eval_local.py`), schema mapped to boolean/enum fields, one pass per case |

Cost of our additions: **5 subagent calls ≈ $0.03** on the session model; the local run is free (≈30 s total on the M5 after load).

## Results

All answerers, over every question they answered:

```
model            agree   n     acc   per-type (noul/score/choice)
----------------------------------------------------------
sol                 31  37   83.8%   24/28 / 4/5 / 3/4
opus                29  37   78.4%   23/28 / 3/5 / 3/4
typesafe (Jev)      20  26   76.9%   16/19 / 2/5 / 2/2
local-qwen3-8b      36  48   75.0%   28/37 / 4/5 / 4/6
```

Strict like-for-like — **only the 26 (case, question) pairs that every model answered**:

```
model            agree   n     acc
--------------------------------------
sol                 23  26   88.5%
opus                21  26   80.8%
typesafe (Jev)      20  26   76.9%
local-qwen3-8b      20  26   76.9%
```

### The two headlines

1. **The local 8B ties Jev on Jev's own curated cases** against the frontier consensus (20/26 each) — at ~$0 marginal cost and ~0.4–0.6 s per case instead of ~$0.0001–$0.0011 per API case. The heavier models (Sol, Opus) still lead by 3–4 questions.
2. **Nobody is close to consensus-accuracy on these cases.** These are the viewer's *deliberately curated disagreement examples* — cases chosen because models diverge from each other and from the reference. Absolute accuracy here (77–88%) is **not comparable** to the published per-workflow figures (Jev 61.7–76.0%): different case selection, and a much harder subset.

### What the per-type split shows

- The local 8B's Noul behaviour is its strength (28/37 = 75.7% on its full set, 16/19 on the overlap) — it matches Jev's Noul agreement exactly (16/19).
- Nobody does well on the Score question (2–4 out of 5) — the "evidence strength" scalar is the hardest item in the catalog.
- Jev answered only 26 of the 48 reference pairs — it **declined the containment tree on two cases** (its disposition "middle" branch doesn't reach containment), while Opus/Sol answered 37 and the local engine answered all 48. Coverage differences are part of what the numbers encode.

## Caveats (read before citing)

- **n = 5 cases / 26 overlap pairs.** Not statistically meaningful. Treat as an anecdote generator, not an estimate.
- **Curated disagreement cases**, so all numbers are depressed relative to the workflow average by construction.
- The reference is *consensus of two frontier models*, which is itself a model opinion — "agreement with consensus" ≠ correctness.
- Latency comparisons here (local ≈0.4–0.6 s/case on 26→48 questions) are not the same workload as the earlier 28-field benchmark; context sizes differ (3.8k–6k chars here).
- One API model only (deepseek-v4.1-flash via subagents at `variant: max`); no sampling variance measured (single run).

## Files

- `evals/extract_typesafe_mini.py` — pull + normalize the viewer data
- `evals/make_subagent_prompts.py` — build the prompt files for the subagent runs
- `evals/record_subagent_answers.py` + `evals/subagent_answers.json` — recorded answers
- `evals/eval_local.py` — local engine path (Qwen3-8B-4bit)
- `evals/score_eval.py` — the table generator (`evals/results/summary.json`)
- `evals/prompts/case_*.txt` — the exact prompts the subagents answered
