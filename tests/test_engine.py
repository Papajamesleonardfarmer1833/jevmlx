import math

import pytest

from openjev.cli import load_preset
from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.fixture(scope="module")
def engine():
    return load_engine(MODEL_ID)


@pytest.mark.slow
def test_collision_field_scores_honestly(engine):
    model, tokenizer = engine
    schema_dict = {
        "action": {
            "type": "enum",
            "description": "The action to take on this payment request",
            "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "APPROVE"],
        },
    }

    # Both choices start with "BLOCK" -> shared first token -> per-choice rows.
    block_ctx = (
        "Payment request flagged by rules: the card was reported stolen this morning, "
        "the shipping address does not match the billing country, and the buyer asked "
        "to send the goods to a reshipping mule. Block the transaction, do not touch the account."
    )
    approve_ctx = (
        "Payment request from a verified long-time customer for a routine invoice. "
        "All fraud checks passed, the device is recognized, and the amount matches "
        "previous orders. Approve it and release the funds."
    )

    for expected, ctx in (("BLOCK_TRANSACTION", block_ctx), ("APPROVE", approve_ctx)):
        schema = StructuredSchema(schema_dict)
        result = run_parallel_generation(model, tokenizer, ctx, schema)

        assert result["parsed_json"]["action"]["value"] == expected

        top = result["field_telemetry"]["action"]["top_choices"]
        probs = [c["probability"] for c in top]
        assert abs(sum(probs) - 1.0) < 1e-3
        # Honest distribution: distinct per-choice probabilities, not the old
        # clamp+uniform-rest pattern.
        assert len(set(probs)) > 1
        # Full raw per-choice score list exposed for calibration (F2).
        assert len(result["field_telemetry"]["action"]["scores"]) == 3


@pytest.mark.slow
def test_chunking_matches_full_batch_and_counts_passes(engine):
    model, tokenizer = engine
    preset = load_preset("fintech_fraud")
    schema = StructuredSchema(preset["schema"])

    full = run_parallel_generation(model, tokenizer, preset["context"], schema)
    chunked = run_parallel_generation(model, tokenizer, preset["context"], schema, max_rows=5)

    assert full["parsed_json"].keys() == chunked["parsed_json"].keys()
    for fname in full["parsed_json"]:
        assert full["parsed_json"][fname]["value"] == chunked["parsed_json"][fname]["value"], fname

    # Rows: 1 per field, +1 extra per choice for colliding fields (shared first
    # token, the B1 rule). Pass count must match ceil(rows / max_rows).
    plan = schema.compile_batch_plan(tokenizer)
    expected_rows = sum(
        len(p["choice_token_lists"])
        if len({t[0] for t in p["choice_token_lists"]}) < len(p["choice_token_lists"])
        else 1
        for p in plan.values()
    )
    assert chunked["sequential_forward_passes"] == math.ceil(expected_rows / 5)
    assert full["sequential_forward_passes"] == 1


@pytest.mark.slow
def test_multi_field_returns_subset(engine):
    """A multi field returns the subset of options whose boolean row passed 0.5."""
    model, tokenizer = engine
    schema_dict = {
        "flags": {
            "type": "multi",
            "description": "every statement that applies to this support request",
            "choices": ["billing_issue", "technical_issue", "account_issue"],
        },
    }
    ctx = (
        "Support request: since this morning the mobile app crashes whenever the "
        "usage dashboard is opened. Reinstalling did not help, other pages load fine."
    )
    schema = StructuredSchema(schema_dict)
    result = run_parallel_generation(model, tokenizer, ctx, schema)

    parsed = result["parsed_json"]
    assert list(parsed) == ["flags"]
    value = parsed["flags"]["value"]
    assert isinstance(value, list)
    assert set(value) <= {"billing_issue", "technical_issue", "account_issue"}
    assert "technical_issue" in value  # the app-crash context is clearly technical

    telemetry = result["field_telemetry"]["flags"]
    assert telemetry["type"] == "multi"
    assert set(telemetry["per_option"]) == {"billing_issue", "technical_issue", "account_issue"}
    assert all(0.0 <= p <= 1.0 for p in telemetry["per_option"].values())
    assert len(telemetry["scores"]) == 3  # p_true per option, in choices order
    selected = value
    if selected:
        assert telemetry["confidence"] == min(telemetry["per_option"][o] for o in selected)
