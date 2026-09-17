import json

import pytest

from openjev.cli import load_preset
from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@pytest.mark.slow
def test_fintech_fraud_decisions():
    model, tokenizer = load_engine(MODEL_ID)
    preset = load_preset("fintech_fraud")
    schema = StructuredSchema(preset["schema"])
    result = run_parallel_generation(model, tokenizer, preset["context"], schema)

    # Every schema key present, every value in its choices.
    for fname, fdef in schema.fields.items():
        assert fname in result["parsed_json"]
        val = result["parsed_json"][fname]["value"]
        expected = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
        assert str(val).lower() in [c.lower() for c in expected], f"{fname}={val!r} not in choices"

    # Real inference ran: telemetry for every field with valid confidences.
    assert set(result["field_telemetry"]) == set(schema.fields)
    for entry in result["field_telemetry"].values():
        assert 0.0 <= entry["confidence"] <= 1.0

    # The assembled JSON serializes.
    assert isinstance(json.dumps(result["parsed_json"]), str)


def test_validate_json_multi_list_semantics():
    """C4a: multi values compared as lists (set equality, no duplicates)."""
    from openjev.engine import _validate_json

    schema = StructuredSchema(
        {"flags": {"type": "multi", "description": "d", "choices": ["a", "b", "c"]}}
    )

    # Valid subset -> schema_match True.
    parsed, valid, _, missing, invalid, match = _validate_json('{"flags": ["a", "c"]}', schema)
    assert valid and match and not missing and not invalid

    # Same items different order -> still valid.
    _, _, _, _, _, match = _validate_json('{"flags": ["c", "a"]}', schema)
    assert match

    # Duplicate items -> rejected.
    _, _, _, _, invalid, match = _validate_json('{"flags": ["a", "a"]}', schema)
    assert not match and invalid

    # Unknown item -> rejected.
    _, _, _, _, invalid, match = _validate_json('{"flags": ["a", "zzz"]}', schema)
    assert not match and invalid

    # Non-list -> rejected.
    _, _, _, _, invalid, match = _validate_json('{"flags": "a"}', schema)
    assert not match and invalid


def test_validate_json_non_object_is_valid_but_no_match():
    """C4b: a parsed JSON that is not a dict -> is_valid_json=True, match=False."""
    from openjev.engine import _validate_json

    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    for text in ("[1, 2, 3]", '"hello"', "42", "null"):
        parsed, valid, _, _, _, match = _validate_json(text, schema)
        assert valid is True, text
        assert match is False, text
    parsed, valid, _, _, _, match = _validate_json("[1, 2, 3]", schema)
    assert parsed == [1, 2, 3]
