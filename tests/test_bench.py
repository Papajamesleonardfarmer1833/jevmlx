"""Tests for `jevmlx bench` (no model runs): preflight, naming, and the
results summarizer. The slow end-to-end test lives in test_engine.py's
module-scoped engine fixture style; here everything runs on fakes.
"""

import json
from pathlib import Path

import pytest

from benchmarks.summarize_results import summarize
from jevmlx.bench import (
    _track_scorer_grid,
    enforce_folder_size,
    machine_tag,
    model_slug,
)

# --- machine tag ------------------------------------------------------------


def test_machine_tag_from_fake_sysctl(monkeypatch):
    """<chip-lowercase>-<ram>gb from sysctl output; marketing noise dropped."""
    import jevmlx.bench as bench

    def fake_sysctl(args):
        if args == ["machdep.cpu.brand_string"]:
            return "Apple M5 Max"
        if args == ["hw.memsize"]:
            return str(128 * 2**30)
        return None

    monkeypatch.setattr(bench, "_sysctl", fake_sysctl)
    assert machine_tag() == "m5max-128gb"


def test_machine_tag_override_wins(monkeypatch):
    import jevmlx.bench as bench

    monkeypatch.setattr(bench, "_sysctl", lambda args: "garbage")
    assert machine_tag("m2pro-32gb") == "m2pro-32gb"


def test_machine_tag_single_chip_word(monkeypatch):
    import jevmlx.bench as bench

    def fake_sysctl(args):
        if args == ["machdep.cpu.brand_string"]:
            return "Apple M4"
        if args == ["hw.memsize"]:
            return str(16 * 2**30)
        return None

    monkeypatch.setattr(bench, "_sysctl", fake_sysctl)
    assert machine_tag() == "m4-16gb"


# --- preflight refusals -------------------------------------------------------


def test_preflight_refuses_on_battery(monkeypatch):
    import jevmlx.bench as bench

    monkeypatch.setattr(
        bench,
        "_probe",
        lambda cmd: "Now drawing from 'Battery'" if cmd == ["pmset", "-g", "batt"] else None,
    )
    with pytest.raises(SystemExit, match="battery"):
        bench.preflight(force=False, machine_override="m4-16gb")


def test_preflight_battery_overridden_by_force(monkeypatch, capsys):
    import jevmlx.bench as bench

    monkeypatch.setattr(
        bench,
        "_probe",
        lambda cmd: "Now drawing from 'Battery'" if cmd == ["pmset", "-g", "batt"] else None,
    )
    tag = bench.preflight(force=True, machine_override="m4-16gb")
    assert tag == "m4-16gb"
    assert "battery" in capsys.readouterr().out.lower()


def test_preflight_refuses_busy_metal(monkeypatch):
    import jevmlx.bench as bench

    def fake_probe(cmd):
        if cmd == ["pmset", "-g", "batt"]:
            return "AC Power"
        return None

    monkeypatch.setattr(bench, "_probe", fake_probe)
    monkeypatch.setattr(bench, "_metal_resident_bytes", lambda: 2 * 2**30)
    with pytest.raises(SystemExit, match="Metal memory"):
        bench.preflight(force=False, machine_override="m4-16gb")


def test_preflight_busy_metal_overridden_by_force(monkeypatch):
    import jevmlx.bench as bench

    monkeypatch.setattr(
        bench, "_probe", lambda cmd: "AC Power" if cmd == ["pmset", "-g", "batt"] else None
    )
    monkeypatch.setattr(bench, "_metal_resident_bytes", lambda: 2 * 2**30)
    assert bench.preflight(force=True, machine_override="m4-16gb") == "m4-16gb"


# --- naming -------------------------------------------------------------------


def test_model_slug():
    assert model_slug("mlx-community/Qwen2.5-0.5B-Instruct-4bit") == (
        "mlx-community--qwen2.5-0.5b-instruct-4bit"
    )


def test_track_scorer_grid_drops_naive_letters():
    grid = _track_scorer_grid(["parallel", "naive_local"], ["trie", "letters"])
    assert ("parallel", "trie") in grid
    assert ("parallel", "letters") in grid
    assert ("naive_local", "trie") in grid
    assert ("naive_local", "letters") not in grid


# --- folder size guard ----------------------------------------------------------


def test_enforce_folder_size_gzips_large_predictions(tmp_path):
    big = Path(tmp_path) / "parallel-trie-bundled"
    big.mkdir()
    (big / "predictions.jsonl").write_text("x" * (6 * 1024 * 1024))
    assert enforce_folder_size(Path(tmp_path)) is True
    assert (big / "predictions.jsonl.gz").exists()
    assert not (big / "predictions.jsonl").exists()


def test_enforce_folder_size_noop_when_small(tmp_path):
    small = Path(tmp_path) / "combo"
    small.mkdir()
    (small / "predictions.jsonl").write_text("tiny")
    assert enforce_folder_size(Path(tmp_path)) is False
    assert (small / "predictions.jsonl").exists()


# --- summarizer -----------------------------------------------------------------


def _write_report(folder: Path, metrics: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.json").write_text(
        json.dumps({"environment": {"chip": "x"}, "metrics": metrics}), encoding="utf-8"
    )


def test_summarize_two_report_files(tmp_path, capsys):
    root = Path(tmp_path)
    combo_a = root / "m5max-128gb--model-a" / "parallel-trie-bundled"
    combo_b = root / "m5max-128gb--model-a" / "parallel-letters-typesafe"
    _write_report(
        combo_a,
        {
            "accuracy": 0.83,
            "case_exact_match": 0.5,
            "balanced_accuracy": {"field1": 0.8, "field2": 0.6},
            "ece_5bin_equal_mass": 0.0412,
            "any_flip_rate": 0.05,
            "perturbation_flip_rate": 0.1,
            "latency_ms_p50": 12.5,
            "n_cases": 24,
        },
    )
    _write_report(
        combo_b,
        {
            "accuracy": 0.9,
            "case_exact_match": 0.75,
            "balanced_accuracy": {"field1": 0.9},
            "ece_5bin_equal_mass": 0.02,
            "latency_ms_p50": 11.0,
            "n_cases": 40,
        },
    )

    summary = summarize(root)
    text = summary.read_text(encoding="utf-8")
    assert "m5max" in text and "model-a" in text
    assert "parallel-trie-bundled" in text or "parallel | trie | bundled" in text
    assert "0.83" in text
    assert "0.9" in text
    # Balanced accuracy mean: (0.8 + 0.6) / 2 = 0.7 for combo a.
    assert "0.7" in text
    printed = capsys.readouterr().out
    assert "SUMMARY" in printed or "wrote" in printed


def test_summarize_skips_folders_without_report(tmp_path):
    root = Path(tmp_path)
    empty = root / "m4-16gb--model-b" / "parallel-trie-bundled"
    empty.mkdir(parents=True)
    summary = summarize(root)
    assert "(no report.json files found" in summary.read_text(encoding="utf-8")


def test_summarize_writes_folder_readme(tmp_path):
    root = Path(tmp_path)
    _write_report(root / "m4-16gb--m" / "parallel-trie-bundled", {"accuracy": 1.0})
    summarize(root)
    assert (root / "README.md").exists()
    assert "jevmlx bench" in (root / "README.md").read_text(encoding="utf-8")
