# Local development

The Space is built for Hugging Face ZeroGPU, but the same code runs locally for
development. Without a GPU it falls back to CPU, so expect slow runs.

```bash
cd hf-space
python -m venv .venv-dev
source .venv-dev/bin/activate
pip install -r requirements.txt
BACKEND=torch python app.py
```

The app then serves at http://127.0.0.1:7860.

## Notes

- `BACKEND=torch` forces the PyTorch engine. Without it, the router in
  `core/engine.py` tries the MLX engine first on Apple Silicon.
- The first run downloads the model weights (about 3 GB for the default
  `Qwen/Qwen2.5-1.5B-Instruct`) into your Hugging Face cache.
- Runs on CPU can take tens of seconds; with the comparison checkbox enabled the
  naive baseline is much slower still.
- Set `MODEL_ID` to try a different checkpoint, for example
  `MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct BACKEND=torch python app.py`.
- `local_dev.md`, `push_to_hub.py`, and `.venv-dev/` are development-only files;
  they are harmless for the Space but not required by it.
