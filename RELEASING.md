# Releasing jevmlx

Steps to cut a release (e.g. `0.1.0`). No publish happens automatically.

## Prerelease checklist

- `pytest -m "not slow"` and `pytest -m slow` pass on your Mac.
- `uv build && uv tool run twine check dist/*` passes.
- Wheel smoke-tested in a clean venv: `jevmlx --version`, `jevmlx doctor --json`.
- **Never publish a version whose README numbers are not backed by files
  under `benchmarks/results/`** — every accuracy/latency claim in the README
  must link to a committed result file.

## Steps

1. Update `CHANGELOG.md`: move `Unreleased` entries under a `## [X.Y.Z] -
   YYYY-MM-DD` heading.
2. Bump `version` in `pyproject.toml`, commit, and push to `main`.
3. Tag and push the tag:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

   The `release-check` workflow builds, runs `twine check`, installs the
   wheel into a clean venv, and runs `--version` / `doctor --json`. Green
   run required before publishing.
4. Publish manually (needs a PyPI token with upload rights):

   ```bash
   uv publish --token <PYPI_TOKEN>
   ```

5. Verify on PyPI: `pip install jevmlx` in a fresh venv, rerun the smoke
   test, and check the rendered long description.
