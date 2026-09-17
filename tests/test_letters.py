"""Letter-slot scoring mode (scoring="letters"): plan validation, boundary
checks, prompt text, and engine scoring with a fake model.

Letters mode lists each field's choices as lettered options in the prompt and
reads the next-token distribution restricted to letter slot tokens at one
position. One row per enum/boolean field, one token per choice — no choice
collisions by construction.
"""

import mlx.core as mx
import pytest

from jevmlx.engine import run_parallel_generation
from jevmlx.schema import SchemaCompileError, StructuredSchema


class LetterTokenizer:
    """Byte-pair-ish fake: structure chars are single tokens, letters are
    single round-trip tokens, words are single tokens. Good default for the
    letters mode."""

    name_or_path = "fake-letters"

    def __init__(self):
        self._vocab: dict[int, str] = {i: chr(i) for i in range(32, 127)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if ch in '{}\n ":,':
                # BPE-style: a space followed by a single uppercase letter is
                # ONE token (' A' -> slot token), mirroring real tokenizers.
                if ch == " " and i + 1 < len(text) and len(text) > i + 1:
                    nxt = text[i + 1]
                    if (
                        nxt.isalpha()
                        and nxt.isupper()
                        and (i + 2 >= len(text) or text[i + 2] in '{}\n ":,_')
                    ):
                        word = " " + nxt
                        tid = 1000 + (sum(ord(c) for c in word) % 50000)
                        self._vocab[tid] = word
                        out.append(tid)
                        i += 2
                        continue
                out.append(ord(ch))
                i += 1
                continue
            if ch == "_":
                out.append(95)
                i += 1
                continue
            j = i
            while j < len(text) and text[j] not in '{}\n ":,_':
                j += 1
            word = text[i:j]
            tid = 1000 + (sum(ord(c) for c in word) % 50000)
            self._vocab[tid] = word
            out.append(tid)
            i = j
        return out

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join(self._vocab.get(t, "") for t in ids)

    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode(messages[0]["content"])

    def __len__(self) -> int:
        return 60000


class FakeModel:
    """Zero-logit model with real KVCache objects (no mlx downloads)."""

    def __init__(self, vocab_size: int = 60000, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.args = type("Args", (), {"vocab_size": vocab_size})()
        self.layers = [None] * n_layers

    def parameters(self):
        return {}

    def __call__(self, tokens, cache=None):
        batch, seq_len = tokens.shape
        if cache is not None:
            for c in cache:
                if hasattr(c, "keys") and c.keys is not None:
                    n_kv, seq_cached, head_dim = (
                        c.keys.shape[1],
                        c.keys.shape[2],
                        c.keys.shape[3],
                    )
                    c.keys = mx.zeros((batch, n_kv, seq_cached + seq_len, head_dim))
                    c.values = mx.zeros((batch, n_kv, seq_cached + seq_len, head_dim))
        return mx.zeros((batch, seq_len, self.vocab_size))


# --- slot validation -------------------------------------------------------


def test_letter_must_be_single_round_trip_token():
    """A tokenizer that splits the ' A' slot form into two tokens (no BPE
    merge) -> SchemaCompileError."""

    class SplittingTokenizer(LetterTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text == " A":
                return [32, 11]  # ' A' did not merge into one slot token
            return super().encode(text, add_special_tokens)

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A_VAL", "B_VAL"]}}
    )
    with pytest.raises(SchemaCompileError, match="'A'"):
        schema.compile_letters_plan(SplittingTokenizer())


def test_letter_must_round_trip():
    """A tokenizer whose 'B' token decodes to something else -> error."""

    class LossyTokenizer(LetterTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text == "B":
                return [22]
            return super().encode(text, add_special_tokens)

        def decode(self, ids, skip_special_tokens=True):
            text = super().decode(ids, skip_special_tokens)
            return text.replace("B", "!")

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A_VAL", "B_VAL"]}}
    )
    with pytest.raises(SchemaCompileError, match="round-trip"):
        schema.compile_letters_plan(LossyTokenizer())


def test_slot_tokens_must_be_distinct():
    """Two letters sharing one token id -> SchemaCompileError.

    The fake decodes statefully (the last encoded word wins), so both 'A'
    and 'B' round-trip individually — only the distinctness check can catch
    the collision. Real tokenizers are effectively stateless, which is why
    the round-trip check usually fires first.
    """

    class CollidingTokenizer(LetterTokenizer):
        def __init__(self):
            super().__init__()
            self._last = None

        def encode(self, text, add_special_tokens=False):
            if text in ("A", "B", " A", " B"):
                self._last = text.strip()
                return [42]
            return super().encode(text, add_special_tokens)

        def decode(self, ids, skip_special_tokens=True):
            if list(ids) == [42]:
                return self._last or ""
            return super().decode(ids, skip_special_tokens)

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A_VAL", "B_VAL"]}}
    )
    with pytest.raises(SchemaCompileError, match="not distinct"):
        schema.compile_letters_plan(CollidingTokenizer())


def test_more_than_16_choices_rejected():
    choices = [f"C{i}" for i in range(17)]
    schema = StructuredSchema({"action": {"type": "enum", "description": "d", "choices": choices}})
    with pytest.raises(SchemaCompileError, match="at most 16"):
        schema.compile_letters_plan(LetterTokenizer())


# --- boundary check --------------------------------------------------------


def test_boundary_break_raises_naming_field_and_letter():
    """encode(row + letter) != encode(row) + [slot] -> SchemaCompileError."""

    class FusingTokenizer(LetterTokenizer):
        def encode(self, text, add_special_tokens=False):
            # ': A' fuses into one token, breaking the boundary.
            if text.endswith(": A"):
                base = super().encode(text[:-3], add_special_tokens)
                return [*base, 777]
            return super().encode(text, add_special_tokens)

    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A_VAL", "B_VAL"]}}
    )
    with pytest.raises(SchemaCompileError, match="boundary.*'A'"):
        schema.compile_letters_plan(FusingTokenizer())


def test_boundary_holds_for_every_letter_in_happy_path():
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A_VAL", "B_VAL", "C_VAL"]}}
    )
    plan = schema.compile_letters_plan(LetterTokenizer())
    p = plan["fields"]["action"]
    for letter, slot in zip(p["letters"], p["slot_ids"], strict=True):
        row_prefix = plan["lead_in_ids"] + p["row_ids"]
        joined = LetterTokenizer().encode("{\n" + '  "action": ' + letter)
        assert joined == row_prefix + [slot]


# --- plan text and mapping -------------------------------------------------


def test_plan_text_lists_lettered_choices():
    schema = StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MEDIUM", "HIGH"],
            }
        }
    )
    text = schema.to_letters_schema_str()
    assert '"risk_tier": A) LOW  B) MEDIUM  C) HIGH  // risk level' in text


def test_plan_maps_letters_to_choice_strings_in_schema_order():
    """Position-bias hook: the letter -> choice mapping follows the schema's
    choice order, so eval rotations permute which choice gets which letter."""
    schema = StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MEDIUM", "HIGH"],
            }
        }
    )
    plan = schema.compile_letters_plan(LetterTokenizer())
    assert plan["fields"]["risk_tier"]["choices_map"] == {
        "A": "LOW",
        "B": "MEDIUM",
        "C": "HIGH",
    }
    # A rotated schema assigns letters to the rotated order.
    rotated = StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "risk level",
                "choices": ["MEDIUM", "HIGH", "LOW"],
            }
        }
    )
    rplan = rotated.compile_letters_plan(LetterTokenizer())
    assert rplan["fields"]["risk_tier"]["choices_map"] == {
        "A": "MEDIUM",
        "B": "HIGH",
        "C": "LOW",
    }


def test_booleans_use_letters_for_uniformity():
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    plan = schema.compile_letters_plan(LetterTokenizer())
    p = plan["fields"]["flag"]
    assert p["letters"] == ["A", "B"]
    assert p["choices_map"] == {"A": "true", "B": "false"}


def test_multi_fields_keep_trie_rows():
    schema = StructuredSchema(
        {
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
            "action": {"type": "enum", "description": "d", "choices": ["A_VAL", "B_VAL"]},
        }
    )
    plan = schema.compile_letters_plan(LetterTokenizer())
    assert "row_ids" in plan["fields"]["action"]
    assert "suffix_ids_list" in plan["fields"]["flags"]


# --- engine scoring (fake model) -------------------------------------------


def test_engine_letters_mode_maps_back_to_choice_strings():
    """Uniform logits -> uniform slot distribution; the winner maps back to
    the CHOICE string, not the letter."""
    model = FakeModel()
    tokenizer = LetterTokenizer()
    schema = StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MEDIUM", "HIGH"],
            }
        }
    )
    result = run_parallel_generation(model, tokenizer, "ctx", schema, scoring="letters")

    assert result["confidence_model"] == "letter_slots"
    parsed = result["parsed_json"]["risk_tier"]
    assert parsed["value"] in ("LOW", "MEDIUM", "HIGH")  # a choice string
    assert parsed["prob"] == pytest.approx(1 / 3)
    telemetry = result["field_telemetry"]["risk_tier"]
    assert telemetry["rows"] == 1
    assert set(telemetry["log_scores"]) == {"LOW", "MEDIUM", "HIGH"}


def test_engine_letters_mode_boolean_values():
    model = FakeModel()
    tokenizer = LetterTokenizer()
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    result = run_parallel_generation(model, tokenizer, "ctx", schema, scoring="letters")
    assert result["parsed_json"]["flag"]["value"] in (True, False)
    assert result["confidence_model"] == "letter_slots"


def test_engine_letters_mode_mixed_schema_one_row_per_letter_field():
    """Enum + boolean + multi in letters mode: the multi keeps its option rows."""
    model = FakeModel()
    tokenizer = LetterTokenizer()
    schema = StructuredSchema(
        {
            "risk_tier": {
                "type": "enum",
                "description": "d",
                "choices": ["LOW", "MEDIUM", "HIGH"],
            },
            "flag": {"type": "boolean", "description": "d"},
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    result = run_parallel_generation(model, tokenizer, "ctx", schema, scoring="letters")
    assert set(result["parsed_json"]) == {"risk_tier", "flag", "flags"}
    assert result["field_telemetry"]["risk_tier"]["rows"] == 1
    assert result["field_telemetry"]["flag"]["rows"] == 1
    assert result["field_telemetry"]["flags"]["rows"] == 2
    assert result["confidence_model"] == "letter_slots"


def test_engine_rejects_unknown_scoring_mode():
    model = FakeModel()
    tokenizer = LetterTokenizer()
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    with pytest.raises(ValueError, match="scoring"):
        run_parallel_generation(model, tokenizer, "ctx", schema, scoring="bogus")


def test_engine_trie_mode_default_unchanged():
    """Default stays trie: same call without scoring= behaves as before."""
    model = FakeModel()
    tokenizer = LetterTokenizer()
    schema = StructuredSchema({"flag": {"type": "boolean", "description": "d"}})
    result = run_parallel_generation(model, tokenizer, "ctx", schema)
    assert result["confidence_model"] == "constrained_path"
