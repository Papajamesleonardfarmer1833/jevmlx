# Benchmarks

Evaluation harness for jevmlx. Datasets live as `cases.jsonl` (+ a
`dataset.lock.json` written by the fetchers) and runs write
`predictions.jsonl` + `run.json` into an output directory — one prediction
line per (case, field, permutation), exact contract in the eval-harness
frozen contract doc.

## Tracks

Two tracks answer two different questions.

### Decoder ablation — `parallel` vs `naive_local`

*Same* local model, tokenizer, prompt information, hardware. The only
difference is decoding: jevmlx's parallel constrained path (schema compiled
into a batch plan, every field scored in one pass, log-probabilities straight
off the trie) against the same model free-writing the whole JSON object
(`naive_local`, parsed strictly). Differences here are attributable to
decoding, not weights or serving.

### Product comparison — `parallel` vs `api_baseline`

Local jevmlx against an OpenAI-compatible chat API endpoint (a bigger
remote model, a different serving stack). Reports accuracy, validity, cost,
and end-to-end latency *without attributing the differences to decoding* —
weights, hardware, prompts, and latency boundaries differ by design.

## Commands

```bash
# Decoder ablation, same local model, both tracks:
jevmlx eval --data benchmarks/cases.jsonl --model <model-id> \
    --track parallel --out runs/parallel
jevmlx eval --data benchmarks/cases.jsonl --model <model-id> \
    --track naive_local --out runs/naive_local

# Product comparison against an API endpoint:
OPENAI_API_KEY=... jevmlx eval --data benchmarks/cases.jsonl \
    --track api_baseline --api-base https://api.example.com/v1 \
    --api-model gpt-4o --api-key-env OPENAI_API_KEY --out runs/api

# Position-bias probes (parallel track only):
jevmlx eval --data benchmarks/cases.jsonl --track parallel \
    --permutations rotations --out runs/rotations   # every cyclic rotation of every enum (all k for n<=8, else 8 seeded)
jevmlx eval --data benchmarks/cases.jsonl --track parallel \
    --permutations fieldperm --out runs/fieldperm   # 3 seeded field-order permutations
jevmlx eval --data benchmarks/cases.jsonl --track parallel \
    --permutations all --out runs/all

# Subsets:
jevmlx eval --data benchmarks/cases.jsonl --track parallel \
    --split holdout --out runs/holdout              # train | holdout | all
jevmlx eval --data benchmarks/cases.jsonl --track parallel \
    --limit 20 --out runs/smoke
```

## Artifacts

- `predictions.jsonl` — one line per (run, case, field, permutation):
  prediction, label, `valid`, `correct` (null when no label; invalid counts
  as wrong), `log_scores` (constrained-path log P at T=1, parallel track),
  confidence/per-option, latency, engine rows/passes, error strings. Malformed
  output is a measurement, never a crash. Naive-local lines additionally
  carry `salvage_prediction` (per-field salvage value) so salvage validity
  can be reported as a diagnostic alongside strict validity.
- `run.json` — run id, `environment()` probe, and config: model, temperature
  (decode at T=1 always), track, dataset path + `dataset.lock.json` sha256,
  sha256 of the tokenizer's chat template, sha256 of the compiled batch plan,
  permutation mode, split.
- `dataset.lock.json` — written by the fetchers next to `cases.jsonl`:
  source URLs, sha256s, fetch dates, parser version, counts. Runs reference
  it by hash so any two Macs can reproduce the same input.

Calibration and routing thresholds are fit on train only; holdout and
`benchmark_only` cases never touch them.
