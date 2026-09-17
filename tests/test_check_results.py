"""Tests for benchmarks/check_results.py: valid folder passes, mismatched
report fails, a line missing a key fails. No MLX, no model."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.check_results import check_folder, main
from jevmlx.evalmetrics import compute_metrics
from jevmlx.evalreport import environment, write_report

FIXTURE = Path(__file__).parent / "fixtures" / "results_ok"


def _record(**overrides):
    base = {
        "run_id": "fix",
        "case_id": "c1",
        "group_id": "g",
        "source": "fix",
        "workflow": None,
        "field": "risk",
        "type": "enum",
        "track": "parallel",
        "model": "fake",
        "permutation": "canonical",
        "label": "HIGH",
        "prediction": "HIGH",
        "valid": True,
        "correct": True,
        "log_scores": {"LOW": -2.0, "HIGH": -0.1},
        "probability": 0.7,
        "per_option": None,
        "latency_ms": 5.0,
        "rows": 2,
        "passes": 1,
        "error": None,
        "salvage_prediction": None,
    }
    base.update(overrides)
    return base


def _write_valid(folder: Path, records=None):
    folder.mkdir(parents=True, exist_ok=True)
    if records is None:
        records = [
            _record(),
            _record(case_id="c2", label="LOW", prediction="HIGH", correct=False, probability=0.3),
        ]
    with open(folder / "predictions.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    run = {
        "run_id": "fix",
        "environment": environment(),
        "config": {
            "model": "fake",
            "track": "parallel",
            "temperature": 1.0,
            "dataset_path": "fix",
            "permutations": "none",
            "split": "all",
            "model_revision": None,
            "quantization": None,
            "prompt_version": None,
        },
        "counts": {"cases": 2, "fields": 2, "prediction_lines": len(records)},
    }
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    (folder / "dataset.lock.json").write_text(json.dumps({"source": "fix"}) + "\n")
    write_report(
        folder / "report.json",
        {"environment": environment(), "metrics": compute_metrics(records)},
    )


def test_valid_folder_passes(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    ok, problems = check_folder(folder)
    assert ok, problems
    assert problems == []


def test_committed_fixture_passes():
    """The fixture committed under tests/fixtures/results_ok is itself valid."""
    ok, problems = check_folder(FIXTURE)
    assert ok, problems


def test_mismatched_report_fails(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    # Corrupt one metric in the committed report.json.
    report = json.loads((folder / "report.json").read_text())
    report["metrics"]["accuracy"] = 0.0
    (folder / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ok, problems = check_folder(folder)
    assert not ok
    assert any("accuracy" in p for p in problems)


def test_line_missing_key_fails(tmp_path):
    folder = tmp_path / "combo"
    records = [_record()]
    # Drop a required key from the only line.
    del records[0]["correct"]
    _write_valid(folder, records=records)
    ok, problems = check_folder(folder)
    assert not ok
    assert any("missing keys" in p and "correct" in p for p in problems)


def test_wrong_type_fails(tmp_path):
    folder = tmp_path / "combo"
    records = [_record(valid="yes")]  # valid must be bool
    _write_valid(folder, records=records)
    ok, problems = check_folder(folder)
    assert not ok
    assert any("valid" in p and "wrong type" in p for p in problems)


def test_missing_dataset_lock_fails(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    (folder / "dataset.lock.json").unlink()
    ok, problems = check_folder(folder)
    assert not ok
    assert any("dataset.lock.json" in p for p in problems)


def test_missing_run_key_fails(tmp_path):
    folder = tmp_path / "combo"
    _write_valid(folder)
    run = json.loads((folder / "run.json").read_text())
    del run["counts"]
    (folder / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    ok, problems = check_folder(folder)
    assert not ok
    assert any("counts" in p for p in problems)


def test_main_exits_1_on_failure(tmp_path, capsys):
    folder = tmp_path / "combo"
    _write_valid(folder)
    report = json.loads((folder / "report.json").read_text())
    report["metrics"]["accuracy"] = 0.99
    (folder / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rc = main([str(folder)])
    captured = capsys.readouterr()
    assert rc == 1
    assert "FAIL" in captured.out


def test_main_exits_0_on_success(tmp_path, capsys):
    folder = tmp_path / "combo"
    _write_valid(folder)
    rc = main([str(folder)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "OK" in captured.out


def test_imports_without_mlx(monkeypatch):
    """The check path must import on a no-MLX CI runner (ubuntu-latest)."""
    import sys

    # Simulate mlx absent by marking the modules as None (import raises).
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    # Force a re-import of the check module's dependencies.
    for mod in list(sys.modules):
        if mod.startswith("jevmlx.") and mod not in ("jevmlx.engine",):
            del sys.modules[mod]
    import importlib

    import benchmarks.check_results  # already imported; re-import deps

    importlib.reload(benchmarks.check_results)
    # compute_metrics / environment must be callable without mlx.
    records = [_record()]
    metrics = benchmarks.check_results.compute_metrics(records)
    assert isinstance(metrics, dict)
