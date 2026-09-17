"""Model compatibility matrix for openjev.

Loads each model, runs the fintech_fraud and support_triage presets through
run_parallel_generation, and prints a markdown table with load status,
schema validity, warm latency, prompt size, and peak GPU memory.

Usage: .venv/bin/python tools/compat.py [model_id ...]
"""

import sys
import time

import mlx.core as mx

from openjev.cli import load_preset
from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/gemma-2-2b-it-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
]

PRESETS = ["fintech_fraud", "support_triage"]


def probe(model_id: str) -> dict:
    row = {"model": model_id, "loads": "n", "presets": "", "latency_ms": "-",
           "prompt_tokens": "-", "peak_mem_gb": "-", "error": ""}
    try:
        mx.metal.reset_peak_memory()  # peak is process-wide; isolate per model
        model, tokenizer = load_engine(model_id)
        row["loads"] = "y"
        presets_ok = []
        latencies = []
        prompt_tokens = 0
        for name in PRESETS:
            preset = load_preset(name)
            schema = StructuredSchema(preset["schema"])
            run_parallel_generation(model, tokenizer, preset["context"], schema)  # warmup
            t0 = time.perf_counter()
            result = run_parallel_generation(model, tokenizer, preset["context"], schema)
            latencies.append(result["elapsed_ms"])
            valid = all(
                str(entry["value"]).lower() in [c.lower() for c in schema.fields[f].choices]
                if schema.fields[f].field_type != "boolean"
                else isinstance(result["parsed_json"][f]["value"], bool)
                for f, entry in result["parsed_json"].items()
            )
            presets_ok.append(f"{name}:{'ok' if valid else 'INVALID'}")
            prompt_tokens += len(tokenizer.encode(preset["context"]))
        row["presets"] = ", ".join(presets_ok)
        row["latency_ms"] = f"{sum(latencies) / len(latencies):.0f}"
        row["prompt_tokens"] = str(prompt_tokens)
        row["peak_mem_gb"] = f"{mx.metal.get_peak_memory() / 1e9:.2f}"
    except Exception as e:
        row["error"] = str(e).splitlines()[0][:60]
    return row


def main() -> None:
    models = sys.argv[1:] or MODELS
    rows = []
    for model_id in models:
        print(f"Probing {model_id} ...", flush=True)
        rows.append(probe(model_id))

    print()
    print("| Model | loads | presets valid | warm latency (ms, avg of 2 presets) | prompt tokens | peak GPU mem (GB) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        if r["loads"] == "y":
            print(f"| `{r['model']}` | y | {r['presets']} | {r['latency_ms']} | {r['prompt_tokens']} | {r['peak_mem_gb']} |")
        else:
            print(f"| `{r['model']}` | n | - | - | - | {r['error']} |")


if __name__ == "__main__":
    main()
