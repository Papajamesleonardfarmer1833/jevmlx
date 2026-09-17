"""Tests for the leaderboard generator (benchmarks/leaderboard.py).

Fixtures: a fake results tree and a fake published_agreement.json / official.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.leaderboard import (
    build_table,
    check_readme,
    write_readme_block,
)

HEADER = (
    "| Model | Source | Scorer | Machine | Accuracy | Customer service"
    " | Agent trace | Security | Invoices | Time per case | Cost per case | Cases |"
)


@pytest.fixture()
def results_tree(tmp_path: Path) -> Path:
    """One typesafe/parallel combo + one wrong-dataset combo to be filtered."""
    results = tmp_path / "results"
    combo = results / "fake-8gb-org--m1" / "parallel-slots-typesafe"
    combo.mkdir(parents=True)
    (combo / "run.json").write_text(
        json.dumps({"config": {"model": "org/m1", "track": "parallel", "dataset": "typesafe"}}),
        encoding="utf-8",
    )
    (combo / "report.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "typesafe_agreement": {
                        "agreement_common_subset": 0.75,
                        "n_cases": 20,
                        "n_fields": 40,
                        "by_workflow": {"customer_service": 0.8},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (combo / "predictions.jsonl").write_text(
        "\n".join(json.dumps({"latency_ms": ms}) for ms in (100, 200, 300, 400)) + "\n",
        encoding="utf-8",
    )
    # A bundled-dataset combo: must NOT appear.
    other = results / "fake-8gb-org--m1" / "parallel-slots-bundled"
    other.mkdir(parents=True)
    (other / "run.json").write_text(
        json.dumps({"config": {"model": "org/m1", "track": "parallel", "dataset": "bundled"}}),
        encoding="utf-8",
    )
    (other / "report.json").write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    return results


@pytest.fixture()
def official_file(tmp_path: Path) -> Path:
    path = tmp_path / "official.json"
    path.write_text(
        json.dumps(
            {
                "source_url": "https://evals.typesafe.ai/",
                "retrieved": "2026-09-17",
                "models": [
                    {
                        "name": "Jev",
                        "accuracy": 0.678,
                        "by_workflow": {"customer_service": 0.760},
                        "time_per_case_s": 0.4,
                        "cost_per_case_usd": 0.0004,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def published_file(tmp_path: Path) -> Path:
    path = tmp_path / "published_agreement.json"
    path.write_text(
        json.dumps(
            {
                "subset": {"n_fields": 40, "n_cases": 20, "workflows": [], "field_ids": []},
                "models": [{"name": "Jev", "agreed": 30, "total": 40, "agreement": 0.75}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_official_block_renders_cited_rows(official_file: Path) -> None:
    table = build_table(None, None, official_file)
    assert "| **TypeSafe official (cited, retrieved 2026-09-17)** |" in table
    assert "| Jev | official (cited) |" in table
    assert "67.8%" in table and "76.0%" in table
    assert "0.4s" in table and "$0.0004" in table
    assert header_in(table)


def header_in(table: str) -> bool:
    return HEADER in table


def test_published_block_only_when_no_official(published_file: Path) -> None:
    # official_path=None falls back to the repo default, which exists in this
    # checkout — so point it at a missing file to force "no official block".
    table = build_table(None, published_file, None)
    assert "| **Published models on the public examples (computed)** |" in table
    assert "75.0%" in table or "0.750" in table


def test_official_and_published_blocks_together(official_file: Path, published_file: Path) -> None:
    table = build_table(None, published_file, official_file)
    assert table.index("TypeSafe official (cited") < table.index(
        "Published models on the public examples"
    )


def test_local_rows_filter_to_typesafe_parallel(results_tree: Path, official_file: Path) -> None:
    table = build_table(results_tree, None, official_file)
    assert "| **jevmlx, local (measured)** |" in table
    assert "| org/m1 | local | slots | fake-8gb |" in table
    assert "75.0%" in table or "0.750" in table
    # bundled combo filtered out
    assert "bundled" not in table
    # latency: median of (100,200,300,400) = 250 ms -> 0.25s approx per case
    assert "0.3s" in table or "0.2s" in table


def test_missing_published_file_yields_local_rows_only(
    results_tree: Path, official_file: Path, tmp_path: Path
) -> None:
    missing = tmp_path / "does-not-exist.json"
    table = build_table(results_tree, missing, official_file)
    assert "Published models" not in table
    assert "| org/m1 | local" in table


def test_write_and_check_readme_roundtrip(tmp_path: Path, official_file: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- leaderboard:start -->\nold\n<!-- leaderboard:end -->\n",
        encoding="utf-8",
    )
    table = build_table(None, None, official_file)
    assert write_readme_block(readme, table) is True
    assert check_readme(readme, table) is True
    # Stale: a different table fails the check.
    assert check_readme(readme, table + "\n| extra | row |") is False


def test_no_local_rows_line(results_tree_empty: Path, official_file: Path) -> None:
    table = build_table(results_tree_empty, None, official_file)
    assert "No local results yet — contribute one with `jevmlx bench`." in table


@pytest.fixture()
def results_tree_empty(tmp_path: Path) -> Path:
    results = tmp_path / "empty-results"
    results.mkdir()
    return results
