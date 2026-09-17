import math

import pytest

from openjev.cli import load_preset
from openjev.engine import load_engine, run_naive_generation, run_parallel_generation
from openjev.schema import StructuredSchema
from openjev.trie import build_trie

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
        # Constrained-path log-probabilities exposed per choice for calibration.
        log_scores = result["field_telemetry"]["action"]["log_scores"]
        assert set(log_scores) == {"BLOCK_TRANSACTION", "BLOCK_USER", "APPROVE"}
        assert all(isinstance(v, float) for v in log_scores.values())


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

    # Rows: one per trie branch point for enum/boolean fields, one per option
    # for multi fields. Pass count must match ceil(rows / max_rows).
    plan = schema.compile_batch_plan(tokenizer)
    expected_rows = sum(
        len(build_trie(p["remainders"])) if "options" not in p else len(p["options"])
        for p in plan["fields"].values()
        if "remainders" in p
    )
    assert chunked["sequential_forward_passes"] == math.ceil(expected_rows / 5)
    assert full["sequential_forward_passes"] == 1


@pytest.mark.slow
def test_trie_winner_stable_under_chunking(engine):
    """T4: per-field winners identical with max_rows=3 vs unchunked (trie rows)."""
    model, tokenizer = engine
    for preset_name in ("fintech_fraud", "support_triage"):
        preset = load_preset(preset_name)
        schema = StructuredSchema(preset["schema"])
        full = run_parallel_generation(model, tokenizer, preset["context"], schema)
        chunked = run_parallel_generation(model, tokenizer, preset["context"], schema, max_rows=3)
        for fname in full["parsed_json"]:
            assert full["parsed_json"][fname]["value"] == chunked["parsed_json"][fname]["value"], (
                preset_name,
                fname,
            )


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
    # Per-option rows score at each option's own true/false divergence: the
    # clearly-technical context must not collapse every option to ~0.5 (the
    # cross-option-prefix bug this test now guards against).
    p_tech = telemetry["per_option"]["technical_issue"]
    p_billing = telemetry["per_option"]["billing_issue"]
    assert p_tech > 0.5 and abs(p_billing - 0.5) > 0.05
    assert "log_scores" not in telemetry  # multi: per_option instead, calibrate skips it
    assert len(telemetry["per_option"]) == 3
    selected = value
    if selected:
        assert telemetry["confidence"] == min(telemetry["per_option"][o] for o in selected)


@pytest.mark.slow
def test_mixed_schema_multi_not_collapsed(engine):
    """R1 slow: boolean + enum + multi on the support-triage context.

    The multi per_option values must not all sit near 0.5 (the duplicated
    lead-in would put the branch at the wrong position) and the multi value
    must be a sensible subset for a clearly-technical request.
    """
    model, tokenizer = engine
    preset = load_preset("support_triage")
    schema_dict = dict(preset["schema"])
    schema_dict["extra_flags"] = {
        "type": "multi",
        "description": "every statement that applies to this ticket",
        "choices": ["technical_issue", "billing_issue", "account_issue"],
    }
    schema = StructuredSchema(schema_dict)
    result = run_parallel_generation(model, tokenizer, preset["context"], schema)

    telemetry = result["field_telemetry"]["extra_flags"]
    per_option = telemetry["per_option"]
    assert per_option["technical_issue"] > 0.5  # the context is clearly technical
    assert any(abs(p - 0.5) > 0.05 for p in per_option.values())
    assert "technical_issue" in result["parsed_json"]["extra_flags"]["value"]


@pytest.mark.slow
def test_naive_generation_returns_parseable_text(engine):
    """R3: run_naive_generation must stay callable after the boundary change."""
    import json as _json

    model, tokenizer = engine
    preset = load_preset("fintech_fraud")
    schema = StructuredSchema(preset["schema"])
    result = run_naive_generation(model, tokenizer, preset["context"], schema, max_tokens=200)

    assert result["mode"] == "naive_autoregressive"
    assert isinstance(result["raw_text"], str) and result["raw_text"].startswith("{")
    # Whatever the model produced, the harness must not crash; parsed_json may
    # be None if the model rambles past 200 tokens, but no exception escapes.
    assert result["is_valid_json"] in (True, False)
    if result["is_valid_json"]:
        _json.loads(result["raw_text"])
