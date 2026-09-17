import json

from openjev.cli import load_preset
from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

MODEL_ID = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


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
