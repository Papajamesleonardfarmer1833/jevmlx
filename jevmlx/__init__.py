"""Parallel Constrained Structured Generation Engine."""

import importlib.metadata

__version__ = importlib.metadata.version("jevmlx")

from jevmlx.api import DEFAULT_MODEL, Decision, decide, decide_many, schema_from_model
from jevmlx.engine import (
    clear_engine_cache,
    load_engine,
    run_naive_generation,
    run_parallel_generation,
)
from jevmlx.schema import FieldDefinition, StructuredSchema

__all__ = [
    "DEFAULT_MODEL",
    "Decision",
    "FieldDefinition",
    "StructuredSchema",
    "clear_engine_cache",
    "decide",
    "decide_many",
    "load_engine",
    "run_parallel_generation",
    "run_naive_generation",
    "schema_from_model",
]
