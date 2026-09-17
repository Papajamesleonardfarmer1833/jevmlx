# BENCHMARKING.md — contributing results from your Apple Silicon Mac

Results directories under `benchmarks/results/` are how this project collects
accuracy evidence. One PR adds one directory. This guide makes those results
trustworthy and comparable: same artifacts, same hygiene, no hand-edited
numbers.

## 1. What we collect

Per combination of model (any `mlx-community` 4-bit instruct model), scorer
(`trie` or `letters`), and dataset (bundled cases, TypeSafe public set built
locally, or perturbed cases), commit one directory holding:

- `predictions.jsonl` — one line per (case, field, permutation)
- `run.json` — run manifest (environment, config, counts)
- `report.json` / `report.md` — offline metrics from `jevmlx report`
- `dataset.lock.json` — provenance of the dataset the run consumed

Raw predictions are required, not just reports: metrics can be recomputed
from them when the scorer changes, case-level bootstrap CIs need the
per-line records, and every published number can be audited back to lines.

## 2. Machine requirements and hygiene

- Apple Silicon Mac (M1 or later), macOS 13+. `jevmlx` refuses other hosts.
- RAM guidance from the measured peak GPU memory (see the README's
  compatibility table): 4-bit models up to ~3B peak around 5 GB — 16 GB
  machines are fine; 7B models peak 6–10.5 GB — use 24 GB+ (or quit
  everything else on 16 GB).
- Quit other GPU-heavy apps (browsers with WebGL, Xcode, games). Do not run
  other agents, models, or benchmarks concurrently — Metal memory is shared.
- Plug in. Run the eval twice and keep the second (warm) run; first runs pay
  one-time shader compilation and memory-allocation costs.

## 3. Exact commands

Verified against `--help` on main. `<machine>` naming rule: lowercase model
of your Mac plus RAM, e.g. `m2pro-32gb`, `m4-16gb`.

```bash
git clone https://github.com/bnsd55/jevmlx.git && cd jevmlx
uv venv .venv && .venv/bin/python -m ensurepip -q 2>/dev/null; uv pip install -e '.[dev]'
uv run pytest -m "not slow" -q   # sanity: tests pass on your machine

# Dataset builders (each writes a dataset.lock.json next to its JSONL):
uv run python -m benchmarks.to_jsonl --out cases.jsonl              # bundled cases
uv run python -m benchmarks.typesafe.fetch --out typesafe-cases.jsonl
uv run python -m benchmarks.perturb --in cases.jsonl \
    --out perturbed-cases.jsonl --variants 3 --seed 0

MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit     # any mlx-community 4-bit instruct model
OUT=benchmarks/results/m2pro-32gb-qwen2.5-7b-4bit-trie-bundled

# Eval (run twice; keep the second run), then report — offline:
uv run jevmlx eval --data cases.jsonl --model $MODEL \
    --track parallel --scoring trie --out $OUT
uv run jevmlx report --predictions $OUT/predictions.jsonl --out $OUT/report.json
```

Repeat with `--scoring letters` (name the directory `...-letters-...`) and
with `--data typesafe-cases.jsonl` / `--data perturbed-cases.jsonl`
(`...-typesafe`, `...-perturbed`). One directory per (model × scorer ×
dataset). `run.json` records the environment (OS, Python, mlx / mlx-lm
versions, machine) automatically — never type it by hand.

## 4. What NOT to commit

- Datasets: bundled cases are already in the repo; TypeSafe data is theirs —
  commit only `dataset.lock.json` (URLs + content hashes), never the JSONL.
- Model weights, HF caches, anything from `~/.cache/`.
- Any file over 5 MB uncompressed. Predictions are usually small; gzip only
  if one isn't, and say so in the PR.

## 5. PR checklist

Use the PR template. In short: the results directory is complete (all five
artifacts), you pasted the summary table from
`python benchmarks/summarize_results.py` (lands with the N1 run; until then
paste the metrics table from `report.md`), you pasted the environment lines
from `run.json`, and no number was hand-edited. If `summarize_results.py`
is absent, say so in the PR — your raw artifacts still stand on their own.

## 6. How results are used

The README compatibility table and `benchmarks/results/README.md` are
regenerated from committed `report.json` files only — never from prose or
screenshots. If a number is not in a `report.json` in a merged results
directory, it does not exist.
