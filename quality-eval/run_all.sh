#!/usr/bin/env bash
# Quality eval driver: run the three cached mlx-lm models one at a time, then analyze.
#
# Usage:
#   ./quality-eval/run_all.sh                 # full run (verbose progress, ~15 min)
#   LIMIT=4 ./quality-eval/run_all.sh         # smoke test on the first 4 cases
#   TAG_SUFFIX=retry ./quality-eval/run_all.sh
#
# One process per model, strictly sequential: each run_eval.py exits (and frees
# unified memory) before the next model loads. Missing models are skipped, never
# downloaded.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/.venv/bin/python"
ARTIFACT="$ROOT/vendor/Qwen-2.5-1B-RLCD"
[ -d "$ARTIFACT" ] || ARTIFACT="$ROOT/source/Qwen-2.5-1B-RLCD"
CASES="$HERE/cases.json"
OUTDIR="$HERE/results"
MODELS=(
  "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  "mlx-community/Qwen2.5-7B-Instruct-4bit"
  "mlx-community/Qwen3-8B-4bit"
)
LIMIT="${LIMIT:-0}"
TAG_SUFFIX="${TAG_SUFFIX:-}"

mkdir -p "$OUTDIR"
LOG="$OUTDIR/run_all.log"
: > "$LOG"

say() { echo "$*" | tee -a "$LOG"; }

say "quality-eval driver $(date '+%Y-%m-%d %H:%M:%S')"
say "python:   $PY"
say "artifact: $ARTIFACT"
say "cases:    $CASES (limit=$LIMIT)"

"$PY" "$HERE/make_cases.py" --out "$CASES" 2>&1 | tee -a "$LOG"

ok_models=0
for model in "${MODELS[@]}"; do
  tag="${model##*/}"
  tag="$(echo "$tag" | tr '[:upper:]' '[:lower:]')"
  if [ "$LIMIT" != "0" ]; then tag="$tag-smoke"; fi
  if [ -n "$TAG_SUFFIX" ]; then tag="$tag-$TAG_SUFFIX"; fi

  cache_dir="$HOME/.cache/huggingface/hub/models--${model//\//--}"
  if [ ! -d "$cache_dir" ]; then
    say "[skip] $model is not in the local HF cache (not downloading)"
    continue
  fi

  say ""
  say "=== $model -> results/$tag.json (limit=$LIMIT) $(date '+%H:%M:%S') ==="
  "$PY" -u "$HERE/run_eval.py" \
      --model "$model" --cases "$CASES" --artifact "$ARTIFACT" \
      --outdir "$OUTDIR" --tag "$tag" --limit "$LIMIT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    ok_models=$((ok_models + 1))
  else
    say "[warn] $model exited with code $rc; continuing"
  fi
done

say ""
say "=== analyze $(date '+%H:%M:%S') ==="
"$PY" "$HERE/analyze.py" --outdir "$OUTDIR" --summary-md "$HERE/SUMMARY.md" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  say "[warn] analyze.py exited with code $rc"
fi

say ""
if [ "$ok_models" -eq 0 ]; then
  say "no model completed; nothing useful to report"
  exit 1
fi
say "done: $ok_models/${#MODELS[@]} models completed; summary in $HERE/SUMMARY.md (log: $LOG)"
