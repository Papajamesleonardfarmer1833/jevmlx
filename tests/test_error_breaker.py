"""A1: error-rate circuit breaker per combo.

When fields_error / fields_total > threshold (default 0.10) AND at least 20
fields have been scored, the combo stops: run.json gets a circuit_breaker
dict with the reason, counts, and first error. check_results reports a
tripped combo as FAIL.

Tests:
- 3 of 10 fields error -> trips at the right moment (after 20+ fields,
  when the rate exceeds 0.10)
- 1 of 20 -> no trip
- --max-error-rate 0 -> never trips
- check_results reports a tripped combo as FAIL with the reason
"""

from __future__ import annotations

import json

from jevmlx import evalrun
from jevmlx.evalmetrics import compute_metrics


def _make_cases(n_cases: int, n_fields: int = 1) -> list[dict]:
    """N cases, each with n_fields enum fields."""
    schema = {}
    for i in range(n_fields):
        schema[f"field_{i}"] = {
            "type": "enum",
            "choices": ["A", "B"],
            "description": f"field {i}",
        }
    return [
        {
            "id": f"case-{i}",
            "group_id": f"case-{i}",
            "schema": schema,
            "context": f"context {i}",
            "labels": {f"field_{j}": "A" for j in range(n_fields)},
            "split": "test",
            "meta": {},
        }
        for i in range(n_cases)
    ]


def _make_decide_fn(error_on_fields: set[str] | None = None):
    """A decide_fn that errors on specific field names and succeeds on others.

    error_on_fields: a set of field names (e.g. {"field_0"}) that return an
    error dict instead of a prediction.
    """
    error_on_fields = error_on_fields or set()

    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        results = {}
        for fname in schema_dict:
            if fname in error_on_fields:
                results[fname] = {
                    "prediction": None,
                    "probability": None,
                    "valid": False,
                    "error": f"simulated_error:{fname}",
                }
            else:
                results[fname] = {
                    "prediction": "A",
                    "probability": 0.9,
                    "valid": True,
                }
        return results

    return decide


def _run_and_get_breaker(tmp_path, decide_fn, cases, max_error_rate=0.10):
    """Run eval and return the circuit_breaker entry from run.json.

    A tripped breaker raises CircuitBreakerTrippedError AFTER run.json is
    written; catch it and read run.json (which carries the breaker dict).
    """
    out_dir = tmp_path / "out"
    try:
        evalrun.run_eval(
            cases,
            decide_fn,
            track="parallel",
            model="fake",
            out_dir=str(out_dir),
            max_error_rate=max_error_rate,
        )
    except evalrun.CircuitBreakerTrippedError:
        pass  # expected when the breaker trips
    run = json.loads((out_dir / "run.json").read_text())
    return run.get("circuit_breaker")


def test_breaker_trips_when_error_rate_exceeds_threshold(tmp_path):
    """3 of 10 fields error -> trips after 20+ fields when rate > 0.10.

    With 25 cases × 1 field each (25 fields total), and every 3rd field
    erroring (8 errors / 25 total = 0.32 > 0.10), the breaker trips once
    20 fields have been scored and the rate exceeds 0.10.
    """
    cases = _make_cases(25, n_fields=1)

    # But we want 3 of 10 to error. Use a decide_fn that errors on every
    # 3rd CALL (not field name).
    call_count = [0]

    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        call_count[0] += 1
        results = {}
        for fname in schema_dict:
            if call_count[0] % 3 == 0:  # every 3rd case errors
                results[fname] = {
                    "prediction": None,
                    "probability": None,
                    "valid": False,
                    "error": f"simulated_error:call_{call_count[0]}",
                }
            else:
                results[fname] = {
                    "prediction": "A",
                    "probability": 0.9,
                    "valid": True,
                }
        return results

    breaker = _run_and_get_breaker(tmp_path, decide, cases)

    assert breaker is not None
    assert isinstance(breaker, dict)
    assert breaker["tripped"] is True
    assert "error rate" in breaker["reason"]
    assert breaker["fields_error"] > 0
    assert breaker["fields_total"] >= 20
    assert breaker["first_error"] is not None


def test_breaker_does_not_trip_when_error_rate_below_threshold(tmp_path):
    """1 of 20 fields errors -> rate 0.05 < 0.10 -> no trip."""
    cases = _make_cases(20, n_fields=1)

    call_count = [0]

    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        call_count[0] += 1
        results = {}
        for fname in schema_dict:
            # Only the FIRST case errors (1 of 20 = 0.05 < 0.10).
            if call_count[0] == 1:
                results[fname] = {
                    "prediction": None,
                    "probability": None,
                    "valid": False,
                    "error": "simulated_error:first",
                }
            else:
                results[fname] = {
                    "prediction": "A",
                    "probability": 0.9,
                    "valid": True,
                }
        return results

    breaker = _run_and_get_breaker(tmp_path, decide, cases)

    # No trip — the breaker value is None (or the infra-breaker string, but
    # no errors tripped it).
    assert breaker is None or (
        isinstance(breaker, str) and "circuit_breaker_tripped" not in breaker
    )


def test_breaker_disabled_when_max_error_rate_zero(tmp_path):
    """--max-error-rate 0 -> never trips, even with 100% error rate."""
    cases = _make_cases(25, n_fields=1)

    # Every field errors.
    def decide(schema_dict, context, constraints=None, oracle_overrides=None):
        results = {}
        for fname in schema_dict:
            results[fname] = {
                "prediction": None,
                "probability": None,
                "valid": False,
                "error": "simulated_error:all",
            }
        return results

    breaker = _run_and_get_breaker(tmp_path, decide, cases, max_error_rate=0.0)

    # Never trips.
    assert breaker is None


def test_check_results_reports_tripped_combo_as_fail(tmp_path):
    """check_results reports a tripped combo as FAIL with the reason."""
    from benchmarks.check_results import check_folder

    out_dir = tmp_path / "combo"
    out_dir.mkdir()

    # Write a run.json with a tripped circuit_breaker.
    run = {
        "run_id": "test",
        "environment": {"machine": "test"},
        "config": {
            "model": "fake",
            "temperature": 1.0,
            "track": "parallel",
            "dataset_path": "test.jsonl",
            "permutations": "none",
            "split": "all",
            "model_revision": None,
            "quantization": None,
            "prompt_version": None,
            "dataset_lock_sha256": None,
        },
        "counts": {"cases": 25, "fields": 20, "prediction_lines": 20},
        "circuit_breaker": {
            "tripped": True,
            "reason": "error rate 0.32 > 0.10",
            "fields_error": 8,
            "fields_total": 25,
            "first_error": "simulated_error:call_3",
        },
    }
    (out_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")

    # Write a minimal predictions.jsonl (1 line).
    (out_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "run_id": "test",
                "case_id": "case-1",
                "group_id": "case-1",
                "source": None,
                "workflow": None,
                "field": "field_0",
                "type": "enum",
                "track": "parallel",
                "model": "fake",
                "permutation": "canonical",
                "label": "A",
                "prediction": None,
                "valid": False,
                "correct": False,
                "log_scores": None,
                "probability": None,
                "per_option": None,
                "latency_ms": None,
                "per_item_end_to_end_ms": None,
                "rows": None,
                "passes": None,
                "error": "simulated_error:call_3",
                "salvage_prediction": None,
                "oracle_prediction": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Write a minimal report.json + timing.json.
    (out_dir / "report.json").write_text(
        json.dumps({"environment": {}, "metrics": compute_metrics([])}), encoding="utf-8"
    )
    (out_dir / "timing.json").write_text(
        json.dumps(
            {
                "calls": 1,
                "median": {
                    "prefill_ms": 1.0,
                    "scoring_ms": 1.0,
                    "assembly_ms": 1.0,
                    "per_item_end_to_end_ms": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    ok, problems = check_folder(out_dir)
    assert not ok, f"expected FAIL, got OK: {problems}"
    assert any("circuit breaker tripped" in p for p in problems), (
        f"expected 'circuit breaker tripped' in problems: {problems}"
    )
