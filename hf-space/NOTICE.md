# NOTICE

## Disclaimer

This Space is an **unofficial demo**. It is not affiliated with, endorsed by, or
sponsored by TypeSafe AI.

## Upstream research code

The following files are copied **verbatim** from the upstream research
repository published at https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD:

- `core/__init__.py`
- `core/engine.py`
- `core/engine_torch.py`
- `core/engine_mlx.py`
- `core/schema.py`
- `core/prompt_builder.py`
- `presets/code_security.json`
- `presets/fintech_fraud.json`
- `presets/high_cardinality_255.json`
- `presets/support_triage.json`

The upstream repository is licensed under the **Apache License, Version 2.0**.
It does not ship a LICENSE file, so the standard Apache-2.0 text is included at
`LICENSE`, credited to the upstream authors.

Everything else in this directory (`app.py`, `README.md`, `requirements.txt`,
`push_to_hub.py`, `local_dev.md`, this notice) is original to this Space.

## Model used by this demo

The default model is https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct,
licensed under Apache-2.0.
