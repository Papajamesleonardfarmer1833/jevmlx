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
        # Structural text ({, newline, quotes, spaces, , : , \n) tokenizes
        # char-wise; only the value words map to special (non-compositional)
        # token sequences.
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
    tok = NonCompositionalTokenizer()
    plan = schema.compile_batch_plan(tok)
    remainders = plan["action"]["remainders"]
    # '  "action": "LOW"'+',\n' -> [7, _QUOTE, comma, newline]
    # '  "action": "LOWER"'+',\n' -> [999, _QUOTE, comma, newline]
    # (LOWER is ONE token; the old character-prefix plan would have produced
    # token 8 (ER) somewhere and missed the terminator.)
    assert remainders == [[7, _QUOTE, 44, 10], [999, _QUOTE, 44, 10]]
    flat = [t for remainder in remainders for t in remainder]
    assert 8 not in flat
    # The shared lead-in is the '{\n  "action": "' structure, kept out of
    # shared_ids as the schema-wide prefix (the engine's prefill tail).
    assert plan["_lead_in_ids"] == tok.encode('{\n  "action": "')
    assert plan["action"]["shared_ids"] == []
    assert 999 not in plan["_lead_in_ids"]


def test_plan_cache_is_per_tokenizer():
    """Y6: one schema, two tokenizers -> two distinct cached plans."""
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]}}
    )
    # Keep the tokenizer instances alive: the cache is weakref-keyed, so a
    # dead tokenizer's entry disappears with it.
    tok_a = NonCompositionalTokenizer()
    tok_b = OtherTokenizer()
    plan_a = schema.compile_batch_plan(tok_a)
    plan_b = schema.compile_batch_plan(tok_b)
    assert plan_a["action"]["remainders"] == [[7, _QUOTE, 44, 10], [999, _QUOTE, 44, 10]]
    assert plan_b["action"]["remainders"] == [[3, _QUOTE, 44, 10], [555, _QUOTE, 44, 10]]
    assert len(schema._plans) == 2
    assert schema.compile_batch_plan(tok_a) is plan_a
    assert schema.compile_batch_plan(tok_b) is plan_b


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

    # Temperature applies ONCE to the final scores (softmax(scores / T));
    # per-branch softmax stays at T=1, so ranking is invariant to T.
    probs_hot = softmax(scores, temperature=2.0)
    raw = [math.exp(e / 2.0) for e in expected]
    z = sum(raw)
    assert probs_hot == pytest.approx([p / z for p in raw], abs=1e-12)
    assert max(range(3), key=probs.__getitem__) == max(range(3), key=probs_hot.__getitem__)


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
    lead_in = plan["_lead_in_ids"]
    shared = plan["quote"]["shared_ids"]
    remainders = plan["quote"]["remainders"]

    bs_quote = chr(92) + chr(34)  # backslash + double quote, the JSON escape
    candidate_text = (
        chr(123)
        + chr(10)
        + "  "
        + chr(34)
        + "quote"
        + chr(34)
        + ": "
        + chr(34)
        + "say "
        + bs_quote
        + "hi"
        + bs_quote
        + chr(34)
        + ","
        + chr(10)
    )
    expected_escaped = tok.encode(candidate_text)
    assert lead_in + shared + remainders[0] == expected_escaped
    # A naive f-string candidate (invalid JSON) would tokenize differently.
    assert tok.encode('{\n  "quote": "say ""hi"""\n') != expected_escaped


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


def test_score_trie_rejects_non_finite_logits():
    """T6: NaN/inf logits raise ValueError naming the branch node."""
    nodes = build_trie([[1], [2]])
    with pytest.raises(ValueError, match="non-finite"):
        score_trie(nodes, 2, lambda node: [float("nan"), 1.0])


def test_score_trie_tiny_temperature_ranking_invariant():
    """T5/T6: temperature applied once to final scores; T=1e-3 keeps ranking."""
    remainders = [[1], [2, 3], [2, 4]]
    nodes = build_trie(remainders)
    tables = [[1.0, 2.0], [0.5, -0.5]]
    by_path = {tuple(n["path"]): v for n, v in zip(nodes, tables, strict=True)}
    scores = score_trie(nodes, 3, lambda node: by_path[tuple(node["path"])])

    probs_t1 = softmax(scores)
    probs_tiny = softmax(scores, temperature=1e-3)
    assert all(math.isfinite(p) for p in probs_t1)
    assert all(math.isfinite(p) for p in probs_tiny)
    winner = max(range(3), key=probs_t1.__getitem__)
    assert winner == max(range(3), key=probs_tiny.__getitem__)


def test_single_choice_enum_scores_one_point_oh():
    """T7: cardinality-1 enum -> P=1.0, no branch rows, no crash."""
    schema = StructuredSchema({"only": {"type": "enum", "description": "d", "choices": ["ONLY"]}})
    tok = NonCompositionalTokenizer()
    plan = schema.compile_batch_plan(tok)
    remainders = plan["only"]["remainders"]
    assert len(remainders) == 1
    nodes = build_trie(remainders)
    assert nodes == []  # single leaf: no branch points
    scores = score_trie(nodes, 1, lambda node: [])
    assert scores == [0.0]
    assert softmax(scores)[0] == pytest.approx(1.0)


def test_mixed_schema_rows_carry_lead_in_exactly_once():
    """R1: every row (enum, boolean, multi) starts with the lead-in exactly once."""
    from openjev.trie import build_trie as _bt

    tok = NonCompositionalTokenizer()
    schema = StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {"type": "enum", "description": "d", "choices": ["LOW", "LOWER"]},
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["opt_a", "opt_b"],
            },
        }
    )
    plan = schema.compile_batch_plan(tok)
    lead_in = plan["_lead_in_ids"]
    assert lead_in, "fake tokenizer must produce a shared lead-in"

    # Assemble the rows exactly like the engine does.
    rows: list[list[int]] = []
    for p in plan.values():
        if not isinstance(p, dict):
            continue
        if "options" in p:
            rows.extend(lead_in + list(s) for s in p["suffix_ids_list"])
        elif "remainders" in p:
            trie_nodes = _bt(p["remainders"])
            rows.extend(lead_in + list(p["shared_ids"]) + list(n["path"]) for n in trie_nodes)

    assert rows, "mixed schema must produce rows"
    for row in rows:
        assert row[: len(lead_in)] == lead_in
        assert row[len(lead_in) : len(lead_in) * 2] != lead_in  # not duplicated
    # And the enum candidate must still round-trip to its full text.
    p = plan["action"]
    full = lead_in + p["shared_ids"] + p["remainders"][0]
    assert full == tok.encode('{\n  "action": "LOW",\n')


def test_multi_option_strict_prefix_pair_rejected():
    """R5: an option whose true/false continuations are prefix-related raises."""
    from openjev.schema import StructuredSchema as _SS

    class PrefixPair(NonCompositionalTokenizer):
        name_or_path = "fake-prefix-pair"

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            base = super().encode(text, add_special_tokens)
            # Make option_a's 'true' candidate a strict token-prefix of its
            # 'false' candidate: encode 'true' as the 'false' candidate minus
            # its last token.
            if ".opt_a" in text and "true" in text:
                false_cand = self.encode(text.replace("true", "false"), add_special_tokens)
                return false_cand[:-1]
            return base

    schema = _SS(
        {
            "flags": {
                "type": "multi",
                "description": "d",
                "choices": ["opt_a", "opt_b"],
            }
        }
    )
    with pytest.raises(ValueError, match="strict"):
        schema.compile_batch_plan(PrefixPair())
