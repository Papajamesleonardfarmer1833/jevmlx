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
    p = plan["categories"]

    # One row per option, stable suffix '  "<field>.<option>": '.
    assert p["options"] == ["billing", "technical"]
    assert len(p["suffix_ids_list"]) == 2
    # Each option row has distinct suffix ids, and both rows share the same
    # true/false token lists (the boolean literals).
    assert p["suffix_ids_list"][0] != p["suffix_ids_list"][1]
    assert len(p["choice_token_lists"]) == 2  # [true, false]
    assert all(len(toks) >= 1 for toks in p["choice_token_lists"])
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
    # Two selected: value is the subset, confidence = min p_true of selected.
    selected, conf = _fold_multi({"a": 0.9, "b": 0.6, "c": 0.2})
    assert selected == ["a", "b"]
    assert conf == pytest.approx(0.6)

    # Empty selection: confidence = min p_false over rejected options.
    selected, conf = _fold_multi({"a": 0.2, "b": 0.4, "c": 0.49})
    assert selected == []
    assert conf == pytest.approx(0.51)  # 1 - 0.49

    # Boundary p_true == threshold selects.
    selected, conf = _fold_multi({"a": 0.5, "b": 0.49})
    assert selected == ["a"]
    assert conf == pytest.approx(0.5)
