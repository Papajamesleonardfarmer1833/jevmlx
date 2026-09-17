"""Typed Pydantic API: decide() maps a BaseModel schema onto parallel
constrained decisions and returns validated, typed results.

    from typing import Literal
    from pydantic import BaseModel, Field
    import openjev

    class Fraud(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")

    d = openjev.decide(Fraud, context)
    d.value        # Fraud(is_fraudulent=True, risk_tier="CRITICAL")
    d.confidence   # {"is_fraudulent": 0.99, "risk_tier": 0.97}
    d.latency_ms
"""

from __future__ import annotations

import dataclasses
import enum
import typing
from collections.abc import Sequence

from pydantic import BaseModel

from openjev.engine import load_engine, run_parallel_generation
from openjev.schema import StructuredSchema

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

_SUPPORTED = "supported field types: bool, Literal[str, ...], enum.Enum with str values"


@dataclasses.dataclass
class Decision[T: BaseModel]:
    value: T
    confidence: dict[str, float]
    latency_ms: float


def _description(name: str, info) -> str:
    return info.description or name.replace("_", " ")


def schema_from_model(model_cls: type[BaseModel]) -> dict:
    """Map a Pydantic model to the engine's schema dict (bool / enum fields)."""
    schema: dict = {}
    for name, info in model_cls.model_fields.items():
        ann = info.annotation
        origin = typing.get_origin(ann)
        if ann is bool:
            schema[name] = {"type": "boolean", "description": _description(name, info)}
        elif origin is typing.Literal:
            schema[name] = {
                "type": "enum",
                "choices": [str(c) for c in typing.get_args(ann)],
                "description": _description(name, info),
            }
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            values = [str(m.value) for m in ann]
            schema[name] = {
                "type": "enum",
                "choices": values,
                "description": _description(name, info),
            }
        else:
            raise TypeError(f"Field '{name}' has unsupported type {ann!r}. {_SUPPORTED}")
    return schema


def _decide_once[T: BaseModel](
    model_cls: type[T],
    context: str,
    engine_model,
    tokenizer,
    schema: StructuredSchema,
    temperature: float,
) -> Decision[T]:
    """Decide one context with a loaded engine and a compiled schema."""
    result = run_parallel_generation(
        engine_model, tokenizer, context, schema, temperature=temperature
    )

    kwargs = {}
    for name, info in model_cls.model_fields.items():
        value = result["parsed_json"][name]["value"]
        ann = info.annotation
        if isinstance(ann, type) and issubclass(ann, enum.Enum):
            value = ann(value)
        kwargs[name] = value

    return Decision(
        value=model_cls(**kwargs),
        confidence={k: v["confidence"] for k, v in result["field_telemetry"].items()},
        latency_ms=result["elapsed_ms"],
    )


def decide[T: BaseModel](
    model_cls: type[T],
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
) -> Decision[T]:
    """Run parallel constrained decisions and return a validated model instance."""
    engine_model, tokenizer = load_engine(model)
    schema = StructuredSchema(schema_from_model(model_cls))
    return _decide_once(model_cls, context, engine_model, tokenizer, schema, temperature)


def decide_many[T: BaseModel](
    model_cls: type[T],
    contexts: Sequence[str],
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
) -> list[Decision[T]]:
    """Decide many contexts against one schema and return one Decision per context.

    Loads the model once and compiles the schema once (the engine's batch plan
    is cached on the StructuredSchema instance, so the compiled suffix and
    choice tokens are reused across contexts); the parallel decision pass then
    runs once per context. Results are returned in input order.

    Cross-context batching (all contexts in one forward pass) is future work;
    this is where schema prefix reuse will plug in.
    """
    engine_model, tokenizer = load_engine(model)
    schema = StructuredSchema(schema_from_model(model_cls))
    return [
        _decide_once(model_cls, context, engine_model, tokenizer, schema, temperature)
        for context in contexts
    ]
