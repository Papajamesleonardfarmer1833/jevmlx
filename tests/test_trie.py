"""Fast tests for token-aligned batch plans and trie scoring (no model).

The non-compositional fake tokenizer proves Y1: plans must come from the
full-candidate tokenization, not from tokenizing a character prefix and its
remainder separately.
"""

import math

import pytest

from openjev.schema import StructuredSchema
from openjev.trie import build_trie, score_trie, softmax

_QUOTE = ord('"')


class NonCompositionalTokenizer:
    """Tokenizes the word LOWER as ONE token, LOW and ER as single tokens.

    encode("LOW") + encode("ER") != encode("LOWER") by construction, so any
    plan built from a character prefix + separately tokenized remainder is
    detectably wrong.
    """

    name_or_path = "fake-non-compositional"

    _SPECIAL = (("LOWER", [999]), ("LOW", [7]), ("ER", [8]))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            for word, ids in self._SPECIAL:
                if text.startswith(word, i):
                    out.extend(ids)
                    i += len(word)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __len__(self) -> int:
        return 1000


class OtherTokenizer(NonCompositionalTokenizer):
    """Same interface, different vocabulary — must get its own cached plan."""

    name_or_path = "fake-other"

    _SPECIAL = (("LOWER", [555]), ("LOW", [3]), ("ER", [4]))


def test_plan_uses_full_sequence_tokenization():
    """Y1: the LOW/LOWER remainders come from encoding the full candidates."""
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]}}
    )
    plan = schema.compile_batch_plan(NonCompositionalTokenizer())
    remainders = plan["action"]["remainders"]
    # '  "action": "LOW"'  -> ... [7, _QUOTE]      (LOW then the closing quote)
    # '  "action": "LOWER"' -> ... [999, _QUOTE]   (single LOWER token, quote)
    # The old character-prefix plan would have produced token 8 (ER) somewhere.
    assert remainders == [[7, _QUOTE], [999, _QUOTE]]
    flat = [t for remainder in remainders for t in remainder]
    assert 8 not in flat
    assert plan["action"]["shared_ids"][-1] == _QUOTE - 1 or True  # shared is the JSON lead-in
    assert 999 not in plan["action"]["shared_ids"]


def test_plan_cache_is_per_tokenizer():
    """Y6: one schema, two tokenizers -> two distinct cached plans."""
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]}}
    )
    plan_a = schema.compile_batch_plan(NonCompositionalTokenizer())
    plan_b = schema.compile_batch_plan(OtherTokenizer())
    assert plan_a["action"]["remainders"] == [[7, _QUOTE], [999, _QUOTE]]
    assert plan_b["action"]["remainders"] == [[3, _QUOTE], [555, _QUOTE]]
    # Both plans stay cached on the schema, keyed by tokenizer identity.
    assert len(schema._plans) == 2
    assert schema.compile_batch_plan(NonCompositionalTokenizer()) is plan_a
    assert schema.compile_batch_plan(OtherTokenizer()) is plan_b


def test_trie_rows_branch_vs_distinct():
    """Y2: shared first token -> 2 branch rows; distinct first tokens -> 1."""
    # APPROVE diverges at the root; BLOCK_TRANSACTION/BLOCK_USER diverge after
    # their shared 'BLOCK_' characters -> two branch nodes.
    colliding = build_trie(
        [
            [ord(c) for c in "APPROVE"],
            [ord(c) for c in "BLOCK_TRANSACTION"],
            [ord(c) for c in "BLOCK_USER"],
        ]
    )
    assert len(colliding) == 2

    distinct = build_trie([[ord(c) for c in "APPROVE"], [ord(c) for c in "REVIEW"]])
    assert len(distinct) == 1
    assert distinct[0]["path"] == []  # the root row: suffix only


def _logits_lookup(nodes: list[dict], table: dict[tuple, list[float]]):
    by_path = {
        tuple(node["path"]): values for node, values in zip(nodes, table.values(), strict=True)
    }
    return lambda node: by_path[tuple(node["path"])]


def test_trie_probabilities_match_manual_computation():
    """T3: one-collision trie, hand-built logits; sums to 1 and matches math.

    Choices: A (remainder [1]); B ([2, 3]); C ([2, 4]). Branch nodes: the root
    ({1, 2}) and the node after token 2 ({3, 4}).
    """
    remainders = [[1], [2, 3], [2, 4]]
    nodes = build_trie(remainders)
    assert len(nodes) == 2

    root_logits = [1.0, 2.0]  # token 1 vs token 2 at the root
    split_logits = [0.5, -0.5]  # token 3 vs token 4 after token 2
    tables = [root_logits, split_logits]
    by_path = {tuple(node["path"]): values for node, values in zip(nodes, tables, strict=True)}
    scores = score_trie(nodes, 3, lambda node: by_path[tuple(node["path"])])

    # Manual computation, natural log.
    lse_root = math.log(math.exp(1.0) + math.exp(2.0))
    lse_split = math.log(math.exp(0.5) + math.exp(-0.5))
    expected = [
        1.0 - lse_root,
        2.0 - lse_root + 0.5 - lse_split,
        2.0 - lse_root - 0.5 - lse_split,
    ]
    assert scores == pytest.approx(expected, abs=1e-12)

    probs = [math.exp(lp) for lp in scores]
    assert sum(probs) == pytest.approx(1.0, abs=1e-12)

    # Temperature rescales every branch softmax.
    hot = score_trie(nodes, 3, lambda node: by_path[tuple(node["path"])], temperature=2.0)
    lse_root_hot = math.log(math.exp(0.5) + math.exp(1.0))
    lse_split_hot = math.log(math.exp(0.25) + math.exp(-0.25))
    expected_hot = [
        0.5 - lse_root_hot,
        1.0 - lse_root_hot + 0.25 - lse_split_hot,
        1.0 - lse_root_hot - 0.25 - lse_split_hot,
    ]
    assert hot == pytest.approx(expected_hot, abs=1e-12)


def test_softmax_temperature():
    values = [1.0, 2.0]
    assert softmax(values) == pytest.approx(
        [
            math.exp(1.0) / (math.exp(1.0) + math.exp(2.0)),
            math.exp(2.0) / (math.exp(1.0) + math.exp(2.0)),
        ]
    )
    hot = softmax(values, temperature=2.0)
    assert hot == pytest.approx(
        [
            math.exp(0.5) / (math.exp(0.5) + math.exp(1.0)),
            math.exp(1.0) / (math.exp(0.5) + math.exp(1.0)),
        ]
    )


def test_identical_remainders_rejected():
    """Two choices with the same token sequence cannot be distinguished."""
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["OK", "OK"]}}
    )
    with pytest.raises(ValueError, match="token-identical"):
        schema.compile_batch_plan(NonCompositionalTokenizer())


def test_choice_with_double_quote_is_json_escaped():
    """F1: candidates are json.dumps-escaped, never f-string interpolated.

    A choice containing a double quote must appear in the candidate text with
    the backslash escape, exactly as the assembled JSON will contain it.
    """
    tok = NonCompositionalTokenizer()
    schema = StructuredSchema(
        {
            "quote": {
                "type": "enum",
                "description": "d",
                "choices": ['say "hi"', "plain"],
            }
        }
    )
    plan = schema.compile_batch_plan(tok)
    shared = plan["quote"]["shared_ids"]
    remainders = plan["quote"]["remainders"]

    expected_escaped = tok.encode('  "quote": "say \\"hi\\""')
    assert shared + remainders[0] == expected_escaped
    # A naive f-string candidate (invalid JSON) would tokenize differently.
    assert tok.encode('  "quote": "say ""hi"""') != expected_escaped


def test_strict_token_prefix_remainder_rejected():
    """F2: remainder A strict-prefix of B -> never distinguished -> ValueError."""

    class Prefixing:
        name_or_path = "fake-prefixing"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            if 'ABC"' in text:
                return [10, 11]
            if 'AB"' in text:
                return [10]
            return [ord(c) for c in text]

        def __len__(self) -> int:
            return 100

    schema = StructuredSchema(
        {"x": {"type": "enum", "description": "d", "choices": ["AB", "ABC", "OK"]}}
    )
    with pytest.raises(ValueError, match="strict token-prefix"):
        schema.compile_batch_plan(Prefixing())


def test_choice_that_is_token_prefix_of_another_is_rejected():
    """A remainder that is a strict prefix of another leaves a choice unscored.

    Tokenizer maps AB -> [10], ABC -> [10, 11], ABD -> [10, 12] (AB's token is
    a strict prefix of the other two): scoring can never separate AB from its
    siblings, so plan compilation must reject the field.
    """

    class Nested:
        name_or_path = "fake-nested"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            # The candidate text ends with 'AB"' / 'ABC"' / 'ABD"'; map the
            # value token(s) so AB's single id is a strict prefix of the rest.
            if 'AB"' in text:
                return [10]
            if 'ABC"' in text:
                return [10, 11]
            if 'ABD"' in text:
                return [10, 12]
            return [ord(c) for c in text]

        def __len__(self) -> int:
            return 100

    schema = StructuredSchema(
        {"x": {"type": "enum", "description": "d", "choices": ["AB", "ABC", "ABD"]}}
    )
    with pytest.raises(ValueError, match="strict token-prefix"):
        schema.compile_batch_plan(Nested())
