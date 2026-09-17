"""Fast tests for the multi field type: batch-plan expansion, API mapping,
and the pure fold function. No model loading."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from openjev.api import schema_from_model
from openjev.engine import _fold_multi
from openjev.schema import FieldDefinition, StructuredSchema


class FakeTokenizer:
    """Deterministic tokenizer: one id per character (offset so ids start at 1)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % 97 + 1 for c in text] or [1]


def test_compile_batch_plan_expands_multi_field():
    schema = StructuredSchema(
        {
            "categories": {
                "type": "multi",
                "description": "categories that apply",
                "choices": ["billing", "technical"],
            }
        }
    )
    plan = schema.compile_batch_plan(FakeTokenizer())
    p = plan["fields"]["categories"]

    # One row per option, stable suffix '  "<field>.<option>": '.
    assert p["options"] == ["billing", "technical"]
    assert len(p["suffix_ids_list"]) == 2
    # One row per option, each suffix being that option's shared prefix up to
    # its own true/false divergence.
    assert p["options"] == ["billing", "technical"]
    assert len(p["suffix_ids_list"]) == 2
    assert p["suffix_ids_list"][0] != p["suffix_ids_list"][1]
    # Per-option true/false remainder pairs (2 options x 2), each starting
    # where that option's row ends — the branch compares true vs false there.
    assert len(p["remainders"]) == 2
    assert all(len(pair) == 2 for pair in p["remainders"])
    assert all(len(toks) >= 1 for pair in p["remainders"] for toks in pair)
    assert all(pair[0][0] != pair[1][0] for pair in p["remainders"])
    # Fold mapping: the engine folds rows back by option index.
    assert p["options"].index("billing") == 0
    assert p["options"].index("technical") == 1


def test_multi_field_validation():
    with pytest.raises(ValueError, match="at least 2"):
        FieldDefinition("x", "multi", "d", choices=["only"])
    with pytest.raises(ValueError, match="64"):
        FieldDefinition("x", "multi", "d", choices=[f"c{i}" for i in range(65)])
    with pytest.raises(ValueError, match="multi"):
        FieldDefinition("x", "multi", "d")


def test_schema_from_model_list_and_set_literal():
    class WithTags(BaseModel):
        tags: list[Literal["a", "b"]] = Field(description="tags that apply")
        badges: set[Literal["x", "y"]] = Field(description="badges")

    assert schema_from_model(WithTags) == {
        "tags": {"type": "multi", "choices": ["a", "b"], "description": "tags that apply"},
        "badges": {"type": "multi", "choices": ["x", "y"], "description": "badges"},
    }


def test_schema_from_model_non_literal_list_raises():
    class Bad(BaseModel):
        nums: list[int] = Field(description="numbers")

    with pytest.raises(TypeError, match="'nums'"):
        schema_from_model(Bad)


def test_fold_multi():
    # Two selected; confidence is the min margin over ALL options, so the
    # near-threshold REJECT (c: 1 - 0.2 = 0.8) no longer wins — the weakest
    # selected option does (b: 0.6).
    selected, conf = _fold_multi({"a": 0.9, "b": 0.6, "c": 0.2})
    assert selected == ["a", "b"]
    assert conf == pytest.approx(0.6)

    # A rejected option can be the weakest: c sits just under threshold.
    selected, conf = _fold_multi({"a": 0.99, "b": 0.55, "c": 0.48})
    assert selected == ["a", "b"]
    assert conf == pytest.approx(0.52)  # 1 - 0.48

    # Empty selection: confidence = min (1 - p_true) over rejected options.
    selected, conf = _fold_multi({"a": 0.2, "b": 0.4, "c": 0.49})
    assert selected == []
    assert conf == pytest.approx(0.51)  # 1 - 0.49

    # Boundary p_true == threshold selects.
    selected, conf = _fold_multi({"a": 0.5, "b": 0.49})
    assert selected == ["a"]
    assert conf == pytest.approx(0.5)

    # No options at all: nothing selected, maximally confident.
    selected, conf = _fold_multi({})
    assert selected == []
    assert conf == 1.0
