#!/usr/bin/env bash
# One-command setup: Python env + MLX + the upstream engine (cloned, not vendored).
set -euo pipefail
cd "$(dirname "$0")"

if [ "$(uname -m)" != "arm64" ] || [ "$(uname -s)" != "Darwin" ]; then
  echo "This setup targets Apple Silicon Macs (MLX)."
  echo "On other machines see docs/07-hardware-mac-vs-pc-analysis.md (PyTorch path)."
  exit 1
fi

command -v uv >/dev/null 2>&1 || { echo "uv not found. Install it first: brew install uv"; exit 1; }
command -v git >/dev/null 2>&1 || { echo "git not found. Install Xcode command line tools: xcode-select --install"; exit 1; }

echo "==> Creating Python 3.12 environment (.venv)"
uv venv --python 3.12 .venv

echo "==> Installing mlx-lm (this brings mlx + Metal support)"
uv pip install --python .venv/bin/python mlx-lm

echo "==> Fetching the upstream engine into vendor/ (Apache-2.0, by harshatheg)"
mkdir -p vendor
if [ ! -d vendor/Qwen-2.5-1B-RLCD ]; then
  git clone --depth 1 https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD vendor/Qwen-2.5-1B-RLCD
fi

echo
echo "Done. Next steps:"
echo "  ./run_benchmark.sh                              # default 1.5B model, ~1 min after download"
echo "  ./run_benchmark.sh mlx-community/Qwen2.5-7B-Instruct-4bit"
echo "  .venv/bin/python tools/demo.py                  # single decision example"
