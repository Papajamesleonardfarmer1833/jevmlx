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


def test_decide_many_input_validation(monkeypatch):
    """Y7: str/bytes contexts and non-str items raise TypeError before any load."""

    class TwoField(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(description="Risk tier")

    load_calls: list[str] = []
    monkeypatch.setattr("openjev.api.load_engine", lambda model_id: load_calls.append(model_id))

    with pytest.raises(TypeError, match="sequence of str"):
        openjev.decide_many(TwoField, "one lone context")
    with pytest.raises(TypeError, match="sequence of str"):
        openjev.decide_many(TwoField, b"bytes contexts")
    with pytest.raises(TypeError, match=r"contexts\[1\] must be str"):
        openjev.decide_many(TwoField, ["fine", 123])
    assert load_calls == []  # validation happens before the engine loads


def test_decide_many_empty_contexts_returns_empty_without_loading(monkeypatch):
    """Y7: an empty contexts list short-circuits to [] with no engine load."""

    class TwoField(BaseModel):
        risk_tier: Literal["LOW", "HIGH"] = Field(description="Risk tier")

    def fail_load(model_id):
        raise AssertionError("engine must not be loaded for empty contexts")

    monkeypatch.setattr("openjev.api.load_engine", fail_load)
    assert openjev.decide_many(TwoField, []) == []


def test_schema_from_model_rejects_non_string_literal_values():
    """Y8: Literal[1, 2] must raise, not be silently str()-coerced."""

    class Bad(BaseModel):
        level: Literal[1, 2] = Field(description="level")

    with pytest.raises(TypeError, match="'level'"):
        schema_from_model(Bad)


def test_schema_from_model_rejects_non_string_enum_values():
    """Y8: IntEnum members are not str; must raise naming the field."""

    class Priority(enum.IntEnum):
        LOW = 1
        HIGH = 2

    class Bad(BaseModel):
        priority: Priority = Field(description="priority")

    with pytest.raises(TypeError, match="'priority'"):
        schema_from_model(Bad)


def test_schema_from_model_accepts_strenum():
    """Y8: enum.StrEnum members are str and must work as enum choices."""

    class Color(enum.StrEnum):
        RED = "red"
        GREEN = "green"

    class Ok(BaseModel):
        color: Color = Field(description="color")

    assert schema_from_model(Ok) == {
        "color": {"type": "enum", "choices": ["red", "green"], "description": "color"}
    }


def test_choice_values_rejects_duplicate_values_directly():
    """Y8: the strictness helper rejects duplicates it is handed.

    Tested directly because duplicates cannot reach it through Pydantic:
    Literal deduplicates at annotation level and Python enums alias members
    with equal values, so both collapse before _choice_values runs.
    """
    from openjev.api import _choice_values

    assert _choice_values("x", ["a", "b"]) == ["a", "b"]
    with pytest.raises(TypeError, match="duplicate"):
        _choice_values("x", ["a", "b", "a"])
    with pytest.raises(TypeError, match="non-string"):
        _choice_values("x", ["a", 1])


def test_clear_engine_cache_is_public_and_idempotent():
    """Y9: clear_engine_cache exists on the package and is safe to call twice."""
    assert callable(openjev.clear_engine_cache)
    openjev.clear_engine_cache()
    openjev.clear_engine_cache()  # must not raise with nothing cached


def test_load_engine_cache_is_single_slot():
    """Y9: one model in unified memory at a time — maxsize=1."""
    assert openjev.load_engine.cache_info().maxsize == 1


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
