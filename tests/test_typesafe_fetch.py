"""Tests for the TypeSafe eval fetcher: conversion, split, cache, CLI.

No network: tests run against a synthetic ``__VIEWER_DATA__`` fixture written
to a temp cache dir, mirroring the structure of the published pages.
"""

import json
from pathlib import Path

import pytest

from benchmarks.typesafe.fetch import (
    _viewer_json,
    fetch_all,
    fetch_workflow,
    iter_case_records,
    split_for,
    write_jsonl,
)


def _demo_eval() -> dict:
    """A minimal but structurally faithful eval payload with two cases."""
    return {
        "id": "security_incidents",
        "title": "Security Incidents",
        "documents": [
            "Alert IDS-1001: beaconing observed from workstation-7.",
            {"kind": "log", "lines": ["auth failure x50", "new admin role granted"]},
        ],
        "questions": [
            {
                "type": "noul",
                "instructions": "Is this unauthorized activity?",
                "criteria": {"true": "proof of attack", "false": "authorized"},
            },
            {
                "type": "choice",
                "instructions": "How far does this reach?",
                "criteria": {
                    "single_entity": "One host",
                    "workgroup": "A team",
                    "organization_wide": "Everyone",
                },
            },
            {
                "type": "score",
                "instructions": "How strong is the evidence?",
                "criteria": ["Speculative", "Suggestive"],
            },
            {
                "type": "text",
                "instructions": "Summarize the incident in free text.",
            },
        ],
        "examples": [
            {"case_id": "case-a", "name": "A"},
            {"case_id": "case-b", "name": "B"},
        ],
        "cases": {
            "case-a": {
                "models": {
                    "opus": {
                        "nodes": [
                            {
                                "doc": 0,
                                "questions": {"q_auth": 0, "q_scope": 1, "q_strength": 2},
                                "answers": {
                                    "q_auth": {"type": "noul", "noul": False},
                                    "q_scope": {"type": "choice", "choice": "single_entity"},
                                },
                            }
                        ]
                    },
                    "sol": {
                        "nodes": [
                            {
                                "doc": 0,
                                "answers": {
                                    "q_scope": {"type": "choice", "choice": "organization_wide"}
                                },
                            }
                        ]
                    },
                },
                "reference_answers": {
                    "triage": {
                        "q_auth": {
                            "type": "noul",
                            "sets": [
                                {"value": False, "probabilities": {"true": 0.05, "false": 0.95}},
                                {"value": False, "probabilities": {"true": 0.05, "false": 0.95}},
                            ],
                        },
                        "q_scope": {
                            "type": "choice",
                            "sets": [
                                {
                                    "value": "workgroup",
                                    "probabilities": {
                                        "single_entity": 0.2,
                                        "workgroup": 0.6,
                                        "organization_wide": 0.2,
                                    },
                                },
                                {
                                    "value": "workgroup",
                                    "probabilities": {
                                        "single_entity": 0.1,
                                        "workgroup": 0.8,
                                        "organization_wide": 0.1,
                                    },
                                },
                            ],
                        },
                        "q_strength": {"type": "score", "sets": [{"value": 1}]},
                        "q_summary": {"type": "text", "sets": [{"value": "free text"}]},
                    }
                },
            },
            "case-b": {
                "models": {
                    "opus": {"nodes": [{"doc": 1, "questions": {"q_summary": 3}, "answers": {}}]}
                },
                "reference_answers": {
                    "triage": {"q_summary": {"type": "text", "sets": [{"value": "n/a"}]}}
                },
            },
        },
    }


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    """A cache dir pre-seeded with the synthetic demo workflow page."""
    path = tmp_path / "typesafe"
    path.mkdir(parents=True)
    # The real pages embed {"eval": {...}} (plus viewer chrome) in the call.
    page = "__VIEWER_DATA__(" + json.dumps({"eval": _demo_eval()}) + ");"
    (path / "demo-cases.js").write_text(page, encoding="utf-8")
    return path


def test_viewer_json_extracts_payload():
    payload = _viewer_json('__VIEWER_DATA__({"a": 1});')
    assert payload == {"a": 1}
    with pytest.raises(ValueError, match="__VIEWER_DATA__"):
        _viewer_json("nothing here")


def test_split_is_deterministic():
    """split_for is a pure function of the id."""
    for case_id in ("a", "typesafe/x/y", "z" * 40):
        assert split_for(case_id) == split_for(case_id)


def test_split_holdout_rule_is_documented():
    """holdout iff int(sha1(id) hex[:8], 16) % 5 == 0 — the documented rule."""
    import hashlib

    for case_id in ("typesafe/a/c1", "typesafe/b/c2", "typesafe/c/c3"):
        digest = hashlib.sha1(case_id.encode()).hexdigest()
        expected = "holdout" if int(digest[:8], 16) % 5 == 0 else "train"
        assert split_for(case_id) == expected


def test_iter_case_records_converts_fixture(cache_dir):
    """Two cases -> exact schema fields, labels, per-model meta, skip counts."""
    records = list(iter_case_records("demo", fetch_workflow("demo", cache_dir)))
    assert len(records) == 2

    first = records[0]
    assert first["id"] == "typesafe/demo/case-a"
    assert first["source"] == "typesafe"
    assert first["split"] == split_for("typesafe/demo/case-a")
    # noul -> boolean; choice -> enum with published options; score -> enum 0..n-1.
    assert first["schema"] == {
        "q_auth": {"type": "boolean", "description": "Is this unauthorized activity?"},
        "q_scope": {
            "type": "enum",
            "description": "How far does this reach?",
            "choices": ["single_entity", "workgroup", "organization_wide"],
        },
        "q_strength": {
            "type": "enum",
            "description": "How strong is the evidence? Scale: 0 = Speculative; 1 = Suggestive.",
            "choices": ["0", "1", "2", "3"],
        },
    }
    # q_summary (text) has no node mapping and is skipped.
    assert first["skipped_questions"] == 1
    # Reference labels: averaged probability sets pick the winner, then canonicalize.
    assert first["labels"] == {"q_auth": False, "q_scope": "workgroup", "q_strength": "1"}
    # Per-model published answers, raw as published.
    assert first["meta"]["models"] == {
        "q_auth": {"opus": False},
        "q_scope": {"opus": "single_entity", "sol": "organization_wide"},
    }
    assert "q_strength" not in first["meta"]["models"]
    # Context renders every referenced document, in index order.
    assert first["context"].startswith("## Document 0\nAlert IDS-1001")
    assert "## Document 1" not in first["context"]  # case-a nodes only reference doc 0

    second = records[1]
    # Only a text question, only referenced by node mapping -> nothing decided.
    assert second["id"] == "typesafe/demo/case-b"
    assert second["schema"] == {}
    assert second["labels"] == {}
    assert second["skipped_questions"] == 1
    assert second["context"].startswith("## Document 1\n{")
    assert '"lines"' in second["context"]  # non-string docs are rendered as JSON


def test_fetch_all_summary_counts(cache_dir):
    records, summary = fetch_all(["demo"], cache_dir)
    assert len(records) == 2
    assert "skipped_questions" not in records[0]  # popped into the summary
    assert summary == {
        "workflows": 1,
        "cases": 2,
        "fields": {"boolean": 1, "enum": 2},
        "skipped_questions": 2,
    }


def test_write_jsonl_contract_keys(cache_dir, tmp_path):
    """Each line is a JSON object with exactly the harness contract keys."""
    records, _summary = fetch_all(["demo"], cache_dir)
    out = tmp_path / "cases.jsonl"
    write_jsonl(records, out)
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert lines == records
    for line in lines:
        assert set(line) == {"id", "schema", "context", "labels", "meta", "source", "split"}
        assert line["source"] == "typesafe"
        assert line["split"] in ("train", "holdout")


def test_fetch_workflow_is_offline_when_cached(cache_dir, monkeypatch):
    """With a cached page, fetching never touches the network."""

    def boom(*args, **kwargs):
        raise AssertionError("network access attempted despite cache")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    eval_obj = fetch_workflow("demo", cache_dir)
    assert eval_obj["title"] == "Security Incidents"
    # And again: still no network.
    assert fetch_workflow("demo", cache_dir)["id"] == "security_incidents"


def test_cli_main_writes_jsonl_and_prints_summary(cache_dir, tmp_path, monkeypatch, capsys):
    from benchmarks.typesafe import fetch

    monkeypatch.setattr(fetch, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(fetch, "WORKFLOWS", ("demo",))  # synthetic workflow
    out = tmp_path / "cases.jsonl"
    rc = fetch.main(["--out", str(out), "--workflow", "demo"])
    assert rc == 0
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2
    printed = capsys.readouterr().out
    assert "cases: 2" in printed
    assert "skipped questions (free text): 2" in printed
