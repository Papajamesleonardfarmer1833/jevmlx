"""
Parallel Constrained Decisions - Gradio demo Space.

Unofficial demo: Jev-style typed decisions on a stock Qwen model.
Not affiliated with TypeSafe AI.

The engine itself lives in `core/` (copied verbatim from the upstream research
repository, see NOTICE.md). This file only provides the UI and the ZeroGPU
wiring.

Production runs on Hugging Face Spaces ZeroGPU: the `spaces` package is imported
first so the runtime can patch CUDA, and the function bound to the Gradio event
is decorated with `@spaces.GPU(duration=60)`. When `spaces` is not installed
(local development), the same code degrades to a plain function call.
"""

import json
import os
import traceback

try:
    # ZeroGPU runtime (Hugging Face Spaces). Imported before torch on purpose.
    import spaces

    gpu_decorator = spaces.GPU(duration=60)
except Exception:
    # Local development: no ZeroGPU available, run the decorated function inline.
    def gpu_decorator(fn=None, **kwargs):
        if fn is not None:
            return fn

        return lambda f: f


import gradio as gr

from core.engine import run_naive_generation, run_parallel_generation
from core.schema import StructuredSchema

HERE = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(HERE, "presets")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")

PRESET_ORDER = [
    "fintech_fraud",
    "support_triage",
    "code_security",
    "high_cardinality_255",
]

MAX_ALTERNATIVES = 3

TABLE_HEADERS = ["Field", "Value", "Confidence", "Top alternatives"]
TABLE_DATATYPES = ["str", "str", "number", "str"]

EMPTY_JSON = "{}"
NOT_RUN_TEXT = "_Not run. Enable the comparison checkbox to run the naive JSON baseline._"
INITIAL_LATENCY_TEXT = "_Press Run to see the latency breakdown._"


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

def load_presets():
    """Load every preset JSON shipped in presets/ and index it by preset id."""
    presets = {}
    if not os.path.isdir(PRESETS_DIR):
        return presets

    for fname in sorted(os.listdir(PRESETS_DIR)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(PRESETS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                preset = json.load(handle)
        except Exception:
            traceback.print_exc()
            continue
        key = preset.get("id") or os.path.splitext(fname)[0]
        presets[key] = preset
    return presets


PRESETS = load_presets()


def ordered_preset_keys():
    known = [key for key in PRESET_ORDER if key in PRESETS]
    extra = sorted(key for key in PRESETS if key not in known)
    return known + extra


def preset_label(key):
    preset = PRESETS.get(key, {})
    title = preset.get("title") or key
    return "{} - {}".format(title, key)


def preset_description(key):
    preset = PRESETS.get(key, {})
    lines = []
    if preset.get("title"):
        lines.append("**{}**".format(preset["title"]))
    if preset.get("description"):
        lines.append(preset["description"])
    lines.append("Fields in schema: {}".format(len(preset.get("schema") or {})))
    return "\n\n".join(lines)


def on_preset_change(key):
    preset = PRESETS.get(key, {})
    return preset.get("context", ""), preset_description(key)


# ---------------------------------------------------------------------------
# Rendering helpers (pure functions, no gradio / torch dependency)
# ---------------------------------------------------------------------------

def format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def format_table_rows(result):
    """One row per schema field: value, confidence, runner-up choices."""
    rows = []
    telemetry = (result or {}).get("field_telemetry") or {}
    for name, info in telemetry.items():
        value_text = format_value(info.get("value"))
        confidence = info.get("confidence")
        confidence = float(confidence) if isinstance(confidence, (int, float)) else None

        alternatives = []
        seen = {value_text}
        for choice in (info.get("top_choices") or []):
            choice_text = format_value(choice.get("choice"))
            if choice_text in seen:
                continue
            seen.add(choice_text)
            probability = choice.get("probability")
            try:
                alternatives.append("{} ({:.3f})".format(choice_text, float(probability)))
            except (TypeError, ValueError):
                alternatives.append(choice_text)
            if len(alternatives) >= MAX_ALTERNATIVES:
                break

        rows.append([name, value_text, confidence, ", ".join(alternatives) or "-"])
    return rows


def format_decisions_json(result):
    """The assembled decision document: field name -> chosen value."""
    parsed = (result or {}).get("parsed_json") or {}
    document = {}
    for name, entry in parsed.items():
        if isinstance(entry, dict) and "value" in entry:
            document[name] = entry["value"]
        else:
            document[name] = entry
    return json.dumps(document, indent=2, ensure_ascii=False)


def format_latency(result, naive=None):
    result = result or {}
    lines = [
        "- Prefill: {:.1f} ms".format(float(result.get("prefill_ms") or 0.0)),
        "- Parallel suffix evaluation: {:.1f} ms".format(
            float(result.get("suffix_eval_ms") or 0.0)
        ),
        "- Total: {:.1f} ms".format(float(result.get("elapsed_ms") or 0.0)),
        "- Fields decided in one batched pass: {}".format(result.get("num_fields", 0)),
        "- Sequential decode steps: {} (no token-by-token generation)".format(
            result.get("sequential_forward_passes", 0)
        ),
        "- Device: {}".format(result.get("device", "unknown")),
    ]

    if naive:
        parallel_ms = float(result.get("elapsed_ms") or 0.0)
        naive_ms = float(naive.get("elapsed_ms") or 0.0)
        if parallel_ms > 0.0:
            lines.append(
                "- Naive baseline took {:.1f} ms, so this run was {:.1f}x faster".format(
                    naive_ms, naive_ms / parallel_ms
                )
            )
    return "\n".join(lines)


def format_naive_summary(naive):
    if not naive:
        return NOT_RUN_TEXT

    lines = [
        "- Schema-valid JSON: **{}**".format("yes" if naive.get("is_valid_json") else "no"),
        "- Matches the requested fields: **{}**".format(
            "yes" if naive.get("schema_match") else "no"
        ),
        "- Latency: {:.1f} ms".format(float(naive.get("elapsed_ms") or 0.0)),
        "- Tokens generated: {} ({} tok/s)".format(
            naive.get("total_tokens", 0), naive.get("tokens_per_second", 0)
        ),
        "- Sequential forward passes: {}".format(naive.get("sequential_forward_passes", 0)),
        "- Device: {}".format(naive.get("device", "unknown")),
    ]
    return "\n".join(lines)


def readable_error(exc):
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    hint = ""
    if "gpu task aborted" in lowered or "duration" in lowered or "timeout" in lowered:
        hint = (
            " The model weights may still be downloading on the first request."
            " Wait a few seconds and press Run again."
        )
    return "Engine error: {}{}".format(message, hint)


# ---------------------------------------------------------------------------
# Engine calls (this function is the ZeroGPU entry point wired to the UI)
# ---------------------------------------------------------------------------

@gpu_decorator
def run_engine(context, preset_key, include_naive):
    """Run the parallel engine (and optionally the naive baseline) on the GPU."""
    payload = {"parallel": None, "naive": None, "error": None}
    try:
        text = (context or "").strip()
        if not text:
            payload["error"] = "Please provide some context before running."
            return payload

        preset = PRESETS.get(preset_key) or {}
        schema_dict = preset.get("schema") or {}
        if not schema_dict:
            payload["error"] = "Preset '{}' has no schema.".format(preset_key)
            return payload

        schema = StructuredSchema(schema_dict)
        payload["parallel"] = run_parallel_generation(text, schema)
        if include_naive:
            payload["naive"] = run_naive_generation(text, schema)
    except Exception as exc:
        traceback.print_exc()
        payload["error"] = readable_error(exc)
    return payload


def render_results(payload):
    """Turn the engine payload into the five UI outputs (runs outside the GPU)."""
    try:
        if not payload or payload.get("error") or payload.get("parallel") is None:
            message = (payload or {}).get("error") or "The engine returned no result."
            return [], EMPTY_JSON, message, "", NOT_RUN_TEXT

        parallel = payload["parallel"]
        naive = payload.get("naive")
        return (
            format_table_rows(parallel),
            format_decisions_json(parallel),
            format_latency(parallel, naive),
            (naive or {}).get("raw_text", "") if naive else "",
            format_naive_summary(naive),
        )
    except Exception as exc:
        traceback.print_exc()
        return [], EMPTY_JSON, "Display error: {}".format(exc), "", NOT_RUN_TEXT


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

HEADER = """# Parallel Constrained Decisions

**Unofficial demo: Jev-style typed decisions on a stock Qwen model. Not affiliated with TypeSafe AI.**

Pick a preset, edit the context if you like, then press Run. A small local model
(`{model}`) fills in every field of a typed schema in one batched pass and returns
a value, a calibrated confidence and the runner-up options per field, plus a
latency breakdown. Flip the checkbox to compare against the naive approach, where
the model is asked to write the whole JSON document token by token.
""".format(model=MODEL_ID)

FOOTNOTE = """First run downloads the model weights (about 3 GB), so it can take a while.
Later runs are fast.
"""


def build_demo():
    keys = ordered_preset_keys()
    default_key = keys[0] if keys else None
    default_preset = PRESETS.get(default_key, {})

    with gr.Blocks(title="Parallel Constrained Decisions", theme=gr.themes.Soft()) as demo:
        gr.Markdown(HEADER)

        payload_state = gr.State(None)

        with gr.Row():
            with gr.Column(scale=2):
                preset_dropdown = gr.Dropdown(
                    label="Preset",
                    choices=[(preset_label(key), key) for key in keys],
                    value=default_key,
                )
                preset_info = gr.Markdown(preset_description(default_key))
                context_box = gr.Textbox(
                    label="Context (editable)",
                    value=default_preset.get("context", ""),
                    lines=16,
                    max_lines=30,
                )
                naive_checkbox = gr.Checkbox(
                    label="Also run the naive JSON baseline (slower)",
                    value=False,
                )
                run_button = gr.Button("Run", variant="primary")
                gr.Markdown(FOOTNOTE)

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.Tab("Decisions"):
                        fields_table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=TABLE_DATATYPES,
                            value=[],
                            interactive=False,
                            wrap=True,
                            label="Typed decisions",
                        )
                        decisions_json = gr.Code(
                            value=EMPTY_JSON,
                            language="json",
                            label="Assembled JSON",
                            interactive=False,
                        )
                        latency_markdown = gr.Markdown(INITIAL_LATENCY_TEXT)
                    with gr.Tab("Naive baseline"):
                        naive_text = gr.Code(
                            value="",
                            language="json",
                            label="Raw model output",
                            interactive=False,
                        )
                        naive_markdown = gr.Markdown(NOT_RUN_TEXT)

        preset_dropdown.change(
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[context_box, preset_info],
        )

        run_outputs = [
            fields_table,
            decisions_json,
            latency_markdown,
            naive_text,
            naive_markdown,
        ]

        run_button.click(
            fn=run_engine,
            inputs=[context_box, preset_dropdown, naive_checkbox],
            outputs=[payload_state],
        ).then(fn=render_results, inputs=[payload_state], outputs=run_outputs)

        context_box.submit(
            fn=run_engine,
            inputs=[context_box, preset_dropdown, naive_checkbox],
            outputs=[payload_state],
        ).then(fn=render_results, inputs=[payload_state], outputs=run_outputs)

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.queue().launch()
