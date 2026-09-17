import enum
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import openjev
from openjev.api import schema_from_model
from openjev.cli import load_preset


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
        "is_fraudulent": {"type": "boolean", "description": "Whether the transaction is fraudulent"},
        "risk_tier": {"type": "enum", "choices": ["LOW", "MEDIUM", "HIGH", "CRITICAL"], "description": "Risk tier"},
        "severity": {"type": "enum", "choices": ["LOW", "MEDIUM", "HIGH"], "description": "Case severity"},
        "notes": {"type": "boolean", "description": "notes"},
    }


def test_unsupported_type_raises():
    class Bad(BaseModel):
        count: int = Field(description="how many")

    with pytest.raises(TypeError, match=r"'count'"):
        schema_from_model(Bad)


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
