import enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import openjev
from openjev.api import schema_from_model
from openjev.cli import load_preset
from openjev.schema import StructuredSchema


class Severity(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Case(BaseModel):
    is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")
    severity: Severity = Field(description="Case severity")
    notes: bool  # no description -> field name with underscores replaced


def test_schema_from_model_exact_dict():
    assert schema_from_model(Case) == {
        "is_fraudulent": {
            "type": "boolean",
            "description": "Whether the transaction is fraudulent",
        },
        "risk_tier": {
            "type": "enum",
            "choices": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            "description": "Risk tier",
        },
        "severity": {
            "type": "enum",
            "choices": ["LOW", "MEDIUM", "HIGH"],
            "description": "Case severity",
        },
        "notes": {"type": "boolean", "description": "notes"},
    }


def test_unsupported_type_raises():
    class Bad(BaseModel):
        count: int = Field(description="how many")

    with pytest.raises(TypeError, match=r"'count'"):
        schema_from_model(Bad)


def test_decide_many_uses_one_engine_and_one_schema(monkeypatch):
    class TwoField(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Risk tier")

    load_calls = []
    run_calls = []

    def fake_load_engine(model_id):
        load_calls.append(model_id)
        return ("engine", "tokenizer")

    def fake_run_parallel(engine_model, tokenizer, context, schema, *, temperature=1.0):
        run_calls.append((engine_model, tokenizer, context, schema, temperature))
        return {
            "parsed_json": {
                "is_fraudulent": {"value": True},
                "risk_tier": {"value": "HIGH"},
            },
            "field_telemetry": {
                "is_fraudulent": {"confidence": 0.9},
                "risk_tier": {"confidence": 0.8},
            },
            "elapsed_ms": 5.0,
        }

    monkeypatch.setattr("openjev.api.load_engine", fake_load_engine)
    monkeypatch.setattr("openjev.api.run_parallel_generation", fake_run_parallel)

    decisions = openjev.decide_many(
        TwoField,
        ["context one", "context two", "context three"],
        model="fake/model",
        temperature=0.7,
    )

    assert len(decisions) == 3
    for d in decisions:
        assert d.value == TwoField(is_fraudulent=True, risk_tier="HIGH")
        assert d.confidence == {"is_fraudulent": 0.9, "risk_tier": 0.8}
        assert d.latency_ms == 5.0

    assert load_calls == ["fake/model"]  # engine loaded exactly once
    assert [c[2] for c in run_calls] == ["context one", "context two", "context three"]
    assert all(c[0] == "engine" and c[1] == "tokenizer" for c in run_calls)
    # One shared StructuredSchema object across every context.
    assert len({id(c[3]) for c in run_calls}) == 1
    assert isinstance(run_calls[0][3], StructuredSchema)
    assert all(c[4] == 0.7 for c in run_calls)


def test_decide_end_to_end():
    class TwoField(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")

    fraud_preset = load_preset("fintech_fraud")
    d = openjev.decide(
        TwoField,
        fraud_preset["context"],
        model="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
    )
    assert isinstance(d.value, TwoField)
    assert set(d.confidence) == {"is_fraudulent", "risk_tier"}
    assert all(0.0 <= c <= 1.0 for c in d.confidence.values())
    assert d.latency_ms > 0
