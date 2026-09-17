"""Typed Pydantic API: decide() maps a BaseModel schema onto parallel
constrained decisions and returns validated, typed results.

    from typing import Literal
    from pydantic import BaseModel, Field
    import jevmlx

    class Fraud(BaseModel):
        is_fraudulent: bool = Field(description="Whether the transaction is fraudulent")
        risk_tier: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Risk tier")

    d = jevmlx.decide(Fraud, context)
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

from jevmlx.engine import load_engine, run_parallel_generation
from jevmlx.schema import StructuredSchema

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

_SUPPORTED = (
    "supported field types: bool, Literal[str, ...], enum.Enum/enum.StrEnum with str values, "
    "list[Literal[...]] / set[Literal[...]] (multi)"
)


def _choice_values(name: str, values: list) -> list[str]:
    """Validate and return a field's choice values as strict strings.

    Every value must already be a str — no ``str()`` coercion, which would
    silently turn ``Literal[1, 2]`` into a schema about the strings "1" and
    "2". Duplicates are rejected because the engine scores one row per
    distinct first token and cannot distinguish duplicate literals.
    """
    for value in values:
        if not isinstance(value, str):
            raise TypeError(
                f"Field '{name}' has non-string choice value {value!r} "
                f"({type(value).__name__}); use str values. {_SUPPORTED}"
            )
    if len(set(values)) != len(values):
        dupes = sorted({v for v in values if values.count(v) > 1})
        raise TypeError(
            f"Field '{name}' has duplicate choice values: {', '.join(repr(v) for v in dupes)}"
        )
    return values


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
        elif origin in (list, set) and typing.get_args(ann):
            (lit,) = typing.get_args(ann)
            if typing.get_origin(lit) is not typing.Literal:
                raise TypeError(f"Field '{name}' has unsupported type {ann!r}. {_SUPPORTED}")
            schema[name] = {
                "type": "multi",
                "choices": _choice_values(name, list(typing.get_args(lit))),
                "description": _description(name, info),
            }
        elif origin is typing.Literal:
            schema[name] = {
                "type": "enum",
                "choices": _choice_values(name, list(typing.get_args(ann))),
                "description": _description(name, info),
            }
        elif isinstance(ann, type) and issubclass(ann, enum.Enum):
            values = _choice_values(name, [member.value for member in ann])
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
        origin = typing.get_origin(ann)
        if isinstance(ann, type) and issubclass(ann, enum.Enum):
            value = ann(value)
        elif origin in (list, set):
            value = origin(value)  # list or set of the selected literal strings
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

    Raises:
        TypeError: If ``contexts`` is a bare str or bytes (a common mistake
            that would otherwise be decided one character at a time), or if
            any item is not a str.
    """
    if isinstance(contexts, (str, bytes)):
        raise TypeError(
            f"contexts must be a sequence of str, not {type(contexts).__name__}; "
            "wrap a single context in a list"
        )
    contexts = list(contexts)
    for index, item in enumerate(contexts):
        if not isinstance(item, str):
            raise TypeError(f"contexts[{index}] must be str, not {type(item).__name__}")
    if not contexts:
        return []

    engine_model, tokenizer = load_engine(model)
    schema = StructuredSchema(schema_from_model(model_cls))
    return [
        _decide_once(model_cls, context, engine_model, tokenizer, schema, temperature)
        for context in contexts
    ]
