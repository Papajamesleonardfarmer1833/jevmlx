"""Fetch TypeSafe's public evaluation examples as openjev eval JSONL.

Downloads the published eval viewer data from https://evals.typesafe.ai/
(no JavaScript rendering needed: each workflow ships its cases as a
``<workflow>-cases.js`` file embedding a JSON blob in a
``__VIEWER_DATA__(...)`` call), converts every published case into one
openjev eval JSONL line, and writes them to ``--out``.

Nothing from TypeSafe is committed to this repository; the JSONL produced by
this module is derived data, and raw downloads are cached under
``~/.cache/openjev/typesafe/`` so reruns work offline.

Usage:
    python -m benchmarks.typesafe.fetch --out cases.jsonl [--workflow NAME]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from collections.abc import Iterator
from pathlib import Path

BASE_URL = "https://evals.typesafe.ai"
CACHE_DIR = Path.home() / ".cache" / "openjev" / "typesafe"

# The site rejects requests with the default Python urllib User-Agent (HTTP 403).
USER_AGENT = "Mozilla/5.0 (compatible; openjev-eval-fetcher)"

# The workflows TypeSafe publishes today. A workflow that disappears from the
# site fails loudly on download; pass --workflow to fetch a subset.
WORKFLOWS = (
    "security_incidents",
    "agent_trace_observability",
    "invoice_processing",
    "customer_service",
)

# score questions are answered on a fixed 0-3 scale (their criteria list has
# one description per level).
SCORE_CHOICES = ("0", "1", "2", "3")

_VIEWER_DATA_RE = re.compile(r"__VIEWER_DATA__\((.*)\)\s*;?\s*$", re.S)


def split_for(case_id: str) -> str:
    """Deterministic train/holdout split.

    ``sha1(id)`` interpreted as hex: holdout when the first 8 hex digits as an
    integer are divisible by 5 (~20% holdout), else train. Pure function of the
    id, so the split is stable across runs and machines.
    """
    digest = hashlib.sha1(case_id.encode()).hexdigest()
    return "holdout" if int(digest[:8], 16) % 5 == 0 else "train"


def _viewer_json(raw_js: str) -> dict:
    """Extract the ``__VIEWER_DATA__(<json>)`` payload from a viewer page."""
    match = _VIEWER_DATA_RE.search(raw_js) or re.search(r"__VIEWER_DATA__\((.*)\)", raw_js, re.S)
    if match is None:
        raise ValueError("no __VIEWER_DATA__(...) payload found in page")
    return json.loads(match.group(1))


def _workflow_path(workflow: str, cache_dir: Path) -> Path:
    return cache_dir / f"{workflow}-cases.js"


def fetch_workflow(workflow: str, cache_dir: Path = CACHE_DIR) -> dict:
    """Return the ``eval`` payload for one workflow, using the on-disk cache.

    The raw page is downloaded only when no cached copy exists, so reruns are
    offline. Delete the cached file to force a refresh.
    """
    path = _workflow_path(workflow, cache_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{workflow}-cases.js"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            path.write_bytes(response.read())
    return _viewer_json(path.read_text(encoding="utf-8", errors="replace"))["eval"]


def _canonical_value(value, qtype: str):
    """Canonicalize a published reference value to its openjev label form."""
    if qtype == "noul":
        if isinstance(value, bool):
            return value
        if isinstance(value, int | float):
            return float(value) >= 0.5
        return str(value).lower() == "true"
    if qtype == "score":
        try:
            return str(max(0, min(3, int(round(float(value))))))
        except (TypeError, ValueError):
            return None
    return str(value)


def _consensus(entry: dict) -> tuple[object, dict[str, float]]:
    """Collapse one reference answer's reviewer sets to a value.

    Two published forms: with ``probabilities`` per set (average the
    distributions, take the argmax) or with bare ``value`` per set (majority
    vote, first on tie). Returns ``(label, probability_map)``.
    """
    sets = entry.get("sets") or []
    totals: dict[str, float] = {}
    values: list[object] = []
    for subset in sets:
        for key, prob in (subset.get("probabilities") or {}).items():
            totals[key] = totals.get(key, 0.0) + float(prob)
        if subset.get("value") is not None:
            values.append(subset["value"])
    if totals:
        averaged = {key: total / len(sets) for key, total in totals.items()}
        winner = max(averaged, key=lambda k: averaged[k])
        return _canonical_value(winner, entry["type"]), averaged
    canonical = [
        v for v in (_canonical_value(value, entry["type"]) for value in values) if v is not None
    ]
    if canonical:
        # max over the list (not a set) keeps ties deterministic: first wins.
        return max(canonical, key=canonical.count), {}
    return None, {}


def _model_answers(case: dict, qid: str) -> dict:
    """Per-model published answers for one question: ``{model: raw}``."""
    answers: dict = {}
    for model, model_data in case.get("models", {}).items():
        for node in model_data.get("nodes", []):
            answer = (node.get("answers") or {}).get(qid)
            if answer is None:
                continue
            raw = answer.get(answer.get("type"))
            if raw is not None:
                answers.setdefault(model, raw)
    return answers


def _field_schema(question: dict) -> dict | None:
    """Map one catalog question to an openjev schema field, or None to skip.

    ``noul`` (yes/no) questions become boolean fields. ``choice`` and
    ``score`` questions become enum fields with the published options as
    choices. Anything else (free text) is not decidable by the engine and is
    reported as skipped.
    """
    instructions = question["instructions"]
    criteria = question.get("criteria")
    if question["type"] == "noul":
        return {"type": "boolean", "description": instructions}
    if question["type"] == "choice":
        return {
            "type": "enum",
            "description": instructions,
            "choices": list(criteria),
        }
    if question["type"] == "score":
        levels = "; ".join(f"{i} = {text}" for i, text in enumerate(criteria))
        return {
            "type": "enum",
            "description": f"{instructions} Scale: {levels}.",
            "choices": list(SCORE_CHOICES),
        }
    return None


def _render_doc(doc) -> str:
    return doc if isinstance(doc, str) else json.dumps(doc, indent=1, ensure_ascii=False)


def _case_context(eval_obj: dict, case: dict) -> str:
    """Render every document a case's model nodes reference into one context."""
    doc_indices = {
        int(node["doc"])
        for model_data in case.get("models", {}).values()
        for node in model_data.get("nodes", [])
        if node.get("doc") is not None
    }
    documents = eval_obj["documents"]
    return "\n\n".join(
        f"## Document {i}\n{_render_doc(documents[i])}"
        for i in sorted(doc_indices)
        if 0 <= i < len(documents)
    )


def iter_case_records(workflow: str, eval_obj: dict) -> Iterator[dict]:
    """Yield one eval record per published case of a workflow.

    Each published question becomes a schema field; questions the engine
    cannot decide (free text) are skipped and reflected in the returned
    ``skipped_questions`` count on the record's summary entry.
    """
    catalog = eval_obj["questions"]
    for example in eval_obj["examples"]:
        case_id = example["case_id"]
        case = eval_obj["cases"][case_id]
        case_qmap: dict[str, int] = {}
        for model_data in case.get("models", {}).values():
            for node in model_data.get("nodes", []):
                for qid, idx in (node.get("questions") or {}).items():
                    case_qmap.setdefault(qid, int(idx))

        schema: dict = {}
        labels: dict = {}
        model_meta: dict = {}
        skipped = 0
        for node_answers in case.get("reference_answers", {}).values():
            for qid, entry in node_answers.items():
                field = None
                idx = case_qmap.get(qid)
                if idx is not None and 0 <= idx < len(catalog):
                    field = _field_schema(catalog[idx])
                if field is None:
                    skipped += 1
                    continue
                value, _probs = _consensus(entry)
                if value is None:
                    skipped += 1
                    continue
                schema[qid] = field
                labels[qid] = value
                answers = _model_answers(case, qid)
                if answers:
                    model_meta[qid] = answers

        yield {
            "id": f"typesafe/{workflow}/{case_id}",
            "schema": schema,
            "context": _case_context(eval_obj, case),
            "labels": labels,
            "meta": {"models": model_meta},
            "source": "typesafe",
            "split": split_for(f"typesafe/{workflow}/{case_id}"),
            "skipped_questions": skipped,
        }


def fetch_all(workflows: list[str], cache_dir: Path | None = None) -> tuple[list[dict], dict]:
    """Fetch every workflow and return (records, summary counts)."""
    cache_dir = cache_dir or CACHE_DIR
    records: list[dict] = []
    field_types = {"boolean": 0, "enum": 0}
    skipped_questions = 0
    for workflow in workflows:
        eval_obj = fetch_workflow(workflow, cache_dir)
        for record in iter_case_records(workflow, eval_obj):
            records.append(record)
            skipped_questions += record.pop("skipped_questions")
            for field in record["schema"].values():
                field_types[field["type"]] += 1
    summary = {
        "workflows": len(workflows),
        "cases": len(records),
        "fields": field_types,
        "skipped_questions": skipped_questions,
    }
    return records, summary


def write_jsonl(records: list[dict], out_path: Path) -> None:
    """Write one JSON object per line, the eval harness contract."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.typesafe.fetch",
        description="Fetch TypeSafe's public eval examples as openjev eval JSONL.",
    )
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument(
        "--workflow",
        action="append",
        choices=WORKFLOWS,
        help="fetch only this workflow (repeatable; default: all)",
    )
    args = parser.parse_args(argv)

    workflows = list(args.workflow or WORKFLOWS)
    records, summary = fetch_all(workflows)
    write_jsonl(records, Path(args.out))

    print(f"workflows: {summary['workflows']}")
    print(f"cases: {summary['cases']}")
    print(f"fields: {summary['fields']['boolean']} boolean, {summary['fields']['enum']} enum")
    print(f"skipped questions (free text): {summary['skipped_questions']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
