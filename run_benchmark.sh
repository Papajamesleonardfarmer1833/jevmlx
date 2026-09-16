#!/usr/bin/env bash
# Run the naive-JSON vs parallel-decisions benchmark for one model.
# Usage: ./run_benchmark.sh [hf-model-id]
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${1:-mlx-community/Qwen2.5-1.5B-Instruct-4bit}"
TAG="$(echo "$MODEL" | tr '/' '_')"

echo "Model: $MODEL"
exec .venv/bin/python -u tools/bench_model.py "$MODEL" --tag "$TAG"
