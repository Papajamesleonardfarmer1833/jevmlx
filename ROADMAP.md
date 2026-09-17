# ROADMAP

| Phase | Deliverable |
|---|---|
| P0 | Repo hygiene: openjev name, README, LICENSE+NOTICE, drop x-posts/hf-space/vendor |
| P1 | Package: `openjev/` module, pyproject, `uv pip install -e .`, CLI `openjev decide`, chat template from tokenizer, smoke test |
| P2 | Engine: multi-token collision scoring in one pass, honest confidence, memory auto-chunk |
| P3 | API: Pydantic model in, typed object out |
| P4 | Calibration: `openjev calibrate` fits temperature on labeled JSONL |
| P5 | Public: model compatibility table, CI, PyPI |
