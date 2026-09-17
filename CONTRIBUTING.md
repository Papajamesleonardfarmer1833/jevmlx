# Contributing

## Setup

```bash
git clone https://github.com/bnsd55/openjev && cd openjev
uv venv .venv && uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/pytest -q          # full suite (smoke test downloads a small model)
```

## Rules

- Apple Silicon (Darwin/arm64) only; engine code fails fast elsewhere.
- Branches: short descriptive branch off `main`, pushed, no direct commits to `main`.
- KISS / YAGNI: delete more than you add; no plugin systems, registries, or config files.
- No new dependencies without an issue describing why.
