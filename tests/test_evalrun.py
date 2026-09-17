"""Tests for the eval runner: contract-shaped output from fake decide_fns."""

from __future__ import annotations

import json

from openjev import evalrun


def _two_cases() -> list[dict]:
    return [
        {
            "id": "wf/case-1",
            "group_id": "g1",
            "source": "quality-eval",
            "workflow": None,
            "schema": {
                "action": {
                    "type": "enum",
                    "description": "action to take",
                    "choices": ["APPROVE", "REVIEW", "BLOCK"],
                },
                "flag": {"type": "boolean", "description": "is it urgent"},
            },
            "context": "customer text",
            "labels": {"action": "REVIEW", "flag": True},
            "split": "train",
            "meta": {},
        },
        {
            "id": "wf/case-2",
            "group_id": "g1",
            "source": "typesafe",
            "workflow": "triage",
            "schema": {
                "priority": {
                    "type": "enum",
                    "description": "priority",
                    "choices": ["P1", "P2", "P3"],
                }
            },
            "context": "app crashes",
            "labels": {},
            "split": "holdout",
            "meta": {},
        },
    ]


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_predictions_contract_lines(tmp_path):
    cases = _two_cases()
    run = evalrun.run_eval(
        cases,
        lambda s, c: {
            "action": {"prediction": "APPROVE", "confidence": 0.9},
            "flag": {"prediction": True, "confidence": 0.8},
            "_meta": {"latency_ms": 1.0, "rows": 3, "passes": 1},
        },
        track="parallel",
        model="fake/model",
        out_dir=str(tmp_path),
        run_id="r1",
        chat_template="TEMPLATE",
    )
    assert run["counts"] == {"cases": 2, "fields": 4, "prediction_lines": 4}

    lines = _read_lines(tmp_path / "predictions.jsonl")
    assert len(lines) == 4
    first = lines[0]
    assert set(first) == {
        "run_id", "case_id", "group_id", "source", "workflow", "field", "type",
        "track", "model", "permutation", "label", "prediction", "valid",
        "correct", "log_scores", "confidence", "per_option", "latency_ms",
        "rows", "passes", "error",
    }
    assert first["run_id"] == "r1"
    assert first["track"] == "parallel"
    assert first["permutation"] == "canonical"
    assert first["latency_ms"] == 1.0 and first["rows"] == 3 and first["passes"] == 1
    # correct is (prediction == label) when the label exists, else None
    assert first["correct"] is (first["prediction"] == first["label"])
    holdout = [line for line in lines if line["case_id"] == "wf/case-2"]
    assert all(line["correct"] is None and line["label"] is None for line in holdout)

    run_json = json.load(open(tmp_path / "run.json"))
    assert run_json["run_id"] == "r1"
    assert run_json["config"]["temperature"] == 1.0
    assert run_json["config"]["track"] == "parallel"
    assert len(run_json["config"]["tokenizer_chat_template_sha256"]) == 64
    assert run_json["environment"]["python_version"]


def test_rotation_permutation_remaps_to_canonical(tmp_path):
    case = {
        "id": "c1",
        "group_id": "g1",
        "source": "custom",
        "workflow": None,
        "schema": {
            "risk": {
                "type": "enum",
                "description": "risk level",
                "choices": ["LOW", "MED", "HIGH"],
            }
        },
        "context": "ctx",
        "labels": {"risk": "HIGH"},
        "split": "train",
        "meta": {},
    }
    seen: list[dict] = []

    def fake_decide(schema_dict, context):
        # Always emit the choice that sits FIRST under the current order, so a
        # rotation visibly changes the winner and the remap must map it back.
        seen.append(json.loads(json.dumps(schema_dict)))
        return {"risk": {"prediction": schema_dict["risk"]["choices"][0]}}

    evalrun.run_eval(
        [case],
        fake_decide,
        track="parallel",
        model="fake/model",
        permutations="rotations",
        out_dir=str(tmp_path),
        run_id="r-rot",
        chat_template="T",
        plan_provider=lambda schema: {"compiled": True},
    )
    lines = _read_lines(tmp_path / "predictions.jsonl")
    # canonical + 2 rotations = 3 lines for the single enum field
    assert len(lines) == 3
    tags = {line["permutation"] for line in lines}
    assert tags == {"canonical", "rot1:risk", "rot2:risk"}
    # every remapped prediction is a canonical choice string
    for line in lines:
        assert line["prediction"] in ("LOW", "MED", "HIGH")
    # the rotated schemas the fake actually saw had permuted choices
    assert seen[0]["risk"]["choices"] == ["LOW", "MED", "HIGH"]
    assert seen[1]["risk"]["choices"] == ["MED", "HIGH", "LOW"]
    assert seen[2]["risk"]["choices"] == ["HIGH", "LOW", "MED"]
    # and the canonical winner remaps: rot1 first choice MED -> canonical MED,
    # rot2 first choice HIGH -> canonical HIGH
    by_tag = {line["permutation"]: line["prediction"] for line in lines}
    assert by_tag["canonical"] == "LOW"
    assert by_tag["rot1:risk"] == "MED"
    assert by_tag["rot2:risk"] == "HIGH"
    # correctness is judged against the label AFTER remapping
    assert [line["correct"] for line in lines] == [False, False, True]

    config = json.load(open(tmp_path / "run.json"))["config"]
    assert config["permutations"] == "rotations"
    assert len(config["tokenizer_chat_template_sha256"]) == 64
    assert len(config["compiled_plan_sha256"]) == 64


def test_fieldperm_variants(tmp_path):
    case = {
        "id": "c",
        "group_id": "g",
        "source": "custom",
        "workflow": None,
        "schema": {
            "a": {"type": "boolean", "description": "a"},
            "b": {"type": "boolean", "description": "b"},
            "c": {"type": "boolean", "description": "c"},
        },
        "context": "ctx",
        "labels": {},
        "split": "train",
        "meta": {},
    }
    evalrun.run_eval(
        [case],
        lambda s, c: {name: {"prediction": True} for name in s},
        track="parallel",
        model="m",
        permutations="fieldperm",
        out_dir=str(tmp_path),
    )
    lines = _read_lines(tmp_path / "predictions.jsonl")
    tags = {line["permutation"] for line in lines}
    assert tags == {"canonical", "fieldperm1", "fieldperm2", "fieldperm3"}
    # 4 variants x 3 fields
    assert len(lines) == 12


def test_split_and_dataset_lock(tmp_path):
    cases = [
        {
            "id": f"c{i}",
            "group_id": "g",
            "source": "custom",
            "workflow": None,
            "schema": {"f": {"type": "boolean", "description": "d"}},
            "context": "x",
            "labels": {},
            "split": "holdout" if i % 2 else "train",
            "meta": {},
        }
        for i in range(4)
    ]
    lock = tmp_path / "dataset.lock.json"
    lock.write_text("{}")

    run = evalrun.run_eval(
        cases,
        lambda s, c: {name: {"prediction": True} for name in s},
        track="parallel",
        model="m",
        split="holdout",
        out_dir=str(tmp_path),
        dataset_lock_path=str(lock),
    )
    assert run["counts"]["cases"] == 2
    lines = _read_lines(tmp_path / "predictions.jsonl")
    assert all(line["label"] is None and line["correct"] is None for line in lines)
    config = json.load(open(tmp_path / "run.json"))["config"]
    assert len(config["dataset_lock_sha256"]) == 64


def test_naive_invalid_prediction_counts_wrong(tmp_path):
    """An unparseable naive output yields valid=false, correct=false (label present)."""
    from openjev.baseline import parse_baseline_output
    from openjev.schema import StructuredSchema

    schema = StructuredSchema({"flag": {"type": "boolean", "description": "urgent"}})
    values, errors = parse_baseline_output("not json at all", schema)
    assert values["flag"] is None and errors

    case = {
        "id": "c",
        "group_id": "g",
        "source": "custom",
        "workflow": None,
        "schema": {"flag": {"type": "boolean", "description": "urgent"}},
        "context": "ctx",
        "labels": {"flag": True},
        "split": "train",
        "meta": {},
    }
    evalrun.run_eval(
        [case],
        lambda s, c: {
            "flag": {
                "prediction": None,
                "valid": False,
                "error": "no JSON object found in output",
            }
        },
        track="naive_local",
        model="m",
        out_dir=str(tmp_path),
    )
    line = _read_lines(tmp_path / "predictions.jsonl")[0]
    assert line["valid"] is False
    assert line["correct"] is False  # invalid => wrong when label present
    assert line["error"]


def test_load_cases_skips_comments_and_blank(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "# a comment\n"
        "\n"
        + json.dumps({"id": "c1", "schema": {}, "context": "", "labels": {}})
        + "\n"
    )
    cases = evalrun.load_cases(str(path))
    assert len(cases) == 1 and cases[0]["id"] == "c1"


def test_parallel_log_scores_accessor_prefers_dict(tmp_path, monkeypatch):
    """Accessor reads finalized 'log_scores' (dict) first; falls back to
    legacy 'scores' (list in choice order)."""
    from openjev import evalrun as er

    class FakeField:
        field_type = "enum"
        choices = ["A", "B"]

    class FakeSchema:
        fields = {"x": FakeField()}

    class FakeSchemaCtor:
        def __init__(self, schema_dict):
            pass

        def __getattr__(self, name):
            return FakeSchema().fields  # not used

    # parallel_decide_fn returns decide(); we test decide() by monkeypatching
    # run_parallel_generation inside openjev.engine.
    calls = {}

    def fake_rpg(model, tokenizer, context, schema, temperature=1.0):
        calls["temperature"] = temperature
        return {
            "field_telemetry": {
                "x": {
                    "value": "A",
                    "type": "enum",
                    "confidence": 0.9,
                    "log_scores": {"A": -0.1, "B": -2.0},
                }
            },
            "elapsed_ms": 5.0,
            "rows": 2,
            "passes": 1,
        }

    import openjev.engine as engine_mod

    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg)
    decide = er.parallel_decide_fn(model=object(), tokenizer=object())
    result = decide({"x": {"type": "enum", "description": "d", "choices": ["A", "B"]}}, "ctx")
    assert result["x"]["log_scores"] == {"A": -0.1, "B": -2.0}
    assert calls["temperature"] == 1.0

    # legacy fallback: scores list in choice order
    def fake_rpg_legacy(model, tokenizer, context, schema, temperature=1.0):
        return {
            "field_telemetry": {
                "x": {
                    "value": "B",
                    "type": "enum",
                    "confidence": 0.4,
                    "scores": [-2.0, -0.1],
                }
            },
            "elapsed_ms": 5.0,
            "rows": 2,
            "passes": 1,
        }

    monkeypatch.setattr(engine_mod, "run_parallel_generation", fake_rpg_legacy)
    decide = er.parallel_decide_fn(model=object(), tokenizer=object())
    result = decide({"x": {"type": "enum", "description": "d", "choices": ["A", "B"]}}, "ctx")
    assert result["x"]["log_scores"] == {"A": -2.0, "B": -0.1}
