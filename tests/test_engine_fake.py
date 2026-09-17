"""Engine-level fast tests with a minimal fake model (no mlx downloads).

run_parallel_generation is driven end-to-end on fake logits: prefill and
suffix passes return zeros of the right shape, so branch choices tie at
uniform probability and every row path executes.
"""

import mlx.core as mx
import pytest

from jevmlx.engine import run_parallel_generation
from jevmlx.schema import StructuredSchema


class FakeModel:
    """Minimal model: zeros logits, real KVCache objects sized by layers."""

    def __init__(self, vocab_size: int = 64, n_layers: int = 2):
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.args = type("Args", (), {"vocab_size": vocab_size})()
        self.layers = [None] * n_layers  # make_prompt_cache counts these

    def parameters(self):
        # tree_flatten-able empty tree: zero-weight fake model.
        return {}

    def __call__(self, tokens, cache=None):
        batch, seq_len = tokens.shape
        if cache is not None:
            for c in cache:
                if hasattr(c, "keys") and c.keys is not None:
                    n_kv, seq_cached, head_dim = c.keys.shape[1], c.keys.shape[2], c.keys.shape[3]
                    c.keys = mx.zeros((batch, n_kv, seq_cached + seq_len, head_dim))
                    c.values = mx.zeros((batch, n_kv, seq_cached + seq_len, head_dim))
        return mx.zeros((batch, seq_len, self.vocab_size))


class FakeTokenizer:
    """Character tokenizer with the working-set size readable for the guard."""

    name_or_path = "fake-engine"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) % 60 for c in text]

    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        assert tokenize
        return self.encode(messages[0]["content"])

    def __len__(self) -> int:
        return 64


def test_one_choice_enum_returns_prob_one_without_rows():
    """R2: cardinality-1 enum -> P=1.0 in the engine, no crash, no rows."""
    model = FakeModel()
    tokenizer = FakeTokenizer()
    schema = StructuredSchema({"only": {"type": "enum", "description": "d", "choices": ["ONLY"]}})
    result = run_parallel_generation(model, tokenizer, "ctx", schema)

    assert result["parsed_json"]["only"]["value"] == "ONLY"
    assert result["parsed_json"]["only"]["prob"] == 1.0
    assert result["field_telemetry"]["only"]["rows"] == 0
    assert result["field_telemetry"]["only"]["log_scores"] == {"ONLY": 0.0}


def test_engine_runs_mixed_schema_with_fake_model():
    """R1 smoke at engine level: boolean + enum + multi on fake logits."""
    model = FakeModel()
    tokenizer = FakeTokenizer()
    schema = StructuredSchema(
        {
            "flag": {"type": "boolean", "description": "d"},
            "action": {"type": "enum", "description": "d", "choices": ["A", "B"]},
            "flags": {"type": "multi", "description": "d", "choices": ["x", "y"]},
        }
    )
    result = run_parallel_generation(model, tokenizer, "ctx", schema)

    assert set(result["parsed_json"]) == {"flag", "action", "flags"}
    assert result["confidence_model"] == "constrained_path"
    # Uniform logits -> uniform branch probabilities.
    assert result["parsed_json"]["action"]["prob"] == pytest.approx(0.5)
    telemetry = result["field_telemetry"]["flags"]
    assert all(abs(p - 0.5) < 1e-9 for p in telemetry["per_option"].values())


def test_prompt_sha256_stable_and_input_sensitive():
    """X2: prompt_sha256 is stable for identical inputs and changes when the
    context changes."""
    model = FakeModel()
    tokenizer = FakeTokenizer()
    schema = StructuredSchema(
        {"action": {"type": "enum", "description": "d", "choices": ["A", "B"]}}
    )
    r1 = run_parallel_generation(model, tokenizer, "ctx", schema)
    r2 = run_parallel_generation(model, tokenizer, "ctx", schema)
    r3 = run_parallel_generation(model, tokenizer, "different ctx", schema)

    assert r1["prompt_sha256"] == r2["prompt_sha256"]
    assert r1["prompt_sha256"] != r3["prompt_sha256"]
    assert len(r1["prompt_sha256"]) == 64
    # Independent of the schema contents swap? No: same schema, so identical.
    assert r1["prompt_version"] == "jevmlx-parallel-v1"
    assert (
        r1["probability_status"]
        == "constrained_path probability at T=1; uncalibrated as decision confidence"
    )
