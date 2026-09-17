import math

import pytest

from openjev.cli import load_preset
from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def engine():
    return load_engine(MODEL_ID)


def test_collision_field_scores_honestly(engine):
    model, tokenizer = engine
    schema = StructuredSchema({
        "action": {
            "type": "enum",
            "description": "The action to take on this payment request",
            "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "APPROVE"],
        },
    })
    context = (
        "Payment request from a verified long-time customer for a routine invoice. "
        "All fraud checks passed, the device is recognized, and the amount matches "
        "previous orders. Approve it and release the funds."
    )
    result = run_parallel_generation(model, tokenizer, context, schema)

    assert result["parsed_json"]["action"]["value"] == "APPROVE"

    top = result["field_telemetry"]["action"]["top_choices"]
    probs = [c["probability"] for c in top]
    assert abs(sum(probs) - 1.0) < 1e-3
    # Honest distribution: distinct per-choice probabilities, not the old
    # clamp+uniform-rest pattern.
    assert len(set(probs)) > 1


def test_chunking_matches_full_batch_and_counts_passes(engine):
    model, tokenizer = engine
    preset = load_preset("fintech_fraud")
    schema = StructuredSchema(preset["schema"])

    full = run_parallel_generation(model, tokenizer, preset["context"], schema)
    chunked = run_parallel_generation(model, tokenizer, preset["context"], schema, max_rows=5)

    assert full["parsed_json"].keys() == chunked["parsed_json"].keys()
    for fname in full["parsed_json"]:
        assert full["parsed_json"][fname]["value"] == chunked["parsed_json"][fname]["value"], fname

    # fintech_fraud has 28 non-colliding fields -> 28 rows -> ceil(28/5) chunks.
    assert chunked["sequential_forward_passes"] == math.ceil(28 / 5)
    assert full["sequential_forward_passes"] == 1
