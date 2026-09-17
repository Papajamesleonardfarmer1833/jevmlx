"""Tests for openjev.lint using a deterministic fake tokenizer (no downloads)."""

import zlib

from openjev.lint import lint_schema
from openjev.schema import StructuredSchema


class FakeTokenizer:
    """Maps words to deterministic token ids via crc32.

    One token per underscore-separated word, so choices collide exactly when
    they share their first word and that word is not stripped as the
    schema-wide common prefix.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        return [zlib.crc32(word.encode()) for word in text.split("_")]


def _findings(schema_dict: dict) -> list:
    return lint_schema(StructuredSchema(schema_dict), FakeTokenizer())


def test_collision_detected_with_suggestion():
    """Two choices sharing their first token are flagged, with a rotation rename."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert "BLOCK_TRANSACTION, BLOCK_USER" in collisions[0].message
    assert collisions[0].suggestion == "TRANSACTION_BLOCK, USER_BLOCK"


def test_clean_schema_has_no_findings():
    """Distinct first tokens produce no findings."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "ALLOW", "REVIEW_MANUALLY"],
            }
        }
    )
    assert findings == []


def test_shared_first_word_alone_is_not_a_collision():
    """A shared first word that IS the common prefix is stripped before scoring.

    ["BLOCK_TRANSACTION", "BLOCK_USER"] share the prefix BLOCK; the engine
    strips it and compares TRANSACTION vs USER at the decision position.
    """
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER"],
            }
        }
    )
    assert findings == []


def test_duplicate_choice_flagged():
    """A literal repeated in the choices list gets a duplicate_choice finding."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["ALLOW", "ALLOW", "BLOCK_USER"],
            }
        }
    )
    duplicates = [f for f in findings if f.kind == "duplicate_choice"]
    assert len(duplicates) == 1
    assert 'choice "ALLOW" appears 2 times' in duplicates[0].message


def test_empty_choice_flagged():
    """A choice equal to the shared prefix is scored via the closing quote.

    BLOCK is the common prefix, so it gets the closing-quote row while the
    longer choices keep their own rows — and those longer choices collide on
    the leading-underscore token, so both findings legitimately coexist.
    """
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK", "BLOCK_TRANSACTION", "BLOCK_USER"],
            }
        }
    )
    empties = [f for f in findings if f.kind == "empty_choice"]
    assert len(empties) == 1
    assert 'choice "BLOCK"' in empties[0].message
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert "BLOCK_TRANSACTION, BLOCK_USER" in collisions[0].message


def test_boolean_fields_are_skipped():
    """Booleans always decide true/false; the lint must not touch them."""
    findings = _findings({"approved": {"type": "boolean", "description": "ok?"}})
    assert findings == []


def test_multi_fields_are_skipped():
    """Multi fields decide true/false per option; their plan entry is not per-choice.

    The plan for a multi field carries 'options' and choice_token_lists of
    [true, false], so a per-choice lint loop would index the wrong lists.
    Options are boolean decisions by construction and cannot collide.
    """
    findings = _findings(
        {
            "tags": {
                "type": "multi",
                "description": "select all that apply",
                "choices": ["BLOCK_USER", "BLOCK_TRANSACTION", "ALLOW"],
            }
        }
    )
    assert findings == []


def test_findings_are_dataclasses_with_kind_and_field():
    """Every finding carries field, kind, message; suggestion only when computed."""
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["BLOCK_TRANSACTION", "BLOCK_USER", "ALLOW"],
            }
        }
    )
    assert len(findings) == 1
    finding = findings[0]
    assert finding.field == "action"
    assert finding.kind == "collision"
    assert finding.message
    assert finding.suggestion


def test_collision_without_rotation_suggestion():
    """A collision the rotation rule cannot fix (single-word choice) gets None.

    With no shared prefix, X and X_Y tokenize to [X] and [X, Y] — same first
    token — and X cannot be rotated (single word), so suggestion is None.
    """
    findings = _findings(
        {
            "action": {
                "type": "enum",
                "description": "next action",
                "choices": ["W", "X", "X_Y"],
            }
        }
    )
    collisions = [f for f in findings if f.kind == "collision"]
    assert len(collisions) == 1
    assert "X, X_Y" in collisions[0].message
    assert collisions[0].suggestion is None
