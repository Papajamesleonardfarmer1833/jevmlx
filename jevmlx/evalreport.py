"""Run metadata and human-readable report writer for the eval harness.

Pure, stdlib-only helpers: :func:`environment` probes machine and software
versions, :func:`write_report` renders a run dict to JSON plus a markdown
summary next to it.

Run dict schema (flat and boring; unknown keys pass through to JSON and are
ignored in markdown)::

    {
        "environment": {...},  # see environment()
        "config": {...},       # model id, schema path, temperature, ...
        "metrics": {...},      # accuracy, ece, nll, latency_ms_p50, latency_ms_p90, ...
        "per_field": [{"field": str, "n": int, "accuracy": float}, ...],
        "per_type": {...},     # optional per-type entries (dict or bare accuracy)
    }
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

__all__ = ["environment", "write_report"]


def _probe(command: list[str]) -> str | None:
    """Run a probe command, returning stripped stdout or None on any failure."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _dist_version(name: str) -> str | None:
    """Installed version of a distribution, or None if not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ram_gb() -> float | None:
    """Physical memory in GB (hw.memsize bytes / 2**30), or None off-macOS."""
    raw = _probe(["sysctl", "-n", "hw.memsize"])
    if raw is None:
        return None
    try:
        return round(int(raw) / 2**30, 1)
    except ValueError:
        return None


def _git_sha() -> str | None:
    """HEAD sha when running from a git checkout, else None (e.g. site-packages)."""
    return _probe(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"])


def environment() -> dict:
    """Probe the machine and software versions for an eval run.

    Every field is optional-safe: a failing probe (wrong OS, missing tool,
    uninstalled distribution) degrades to None instead of raising.
    """
    macos = platform.mac_ver()[0] or None
    return {
        "machine_model": _probe(["sysctl", "-n", "hw.model"]),
        "chip": _probe(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "ram_gb": _ram_gb(),
        "macos_version": macos,
        "python_version": sys.version.split()[0],
        "mlx_version": _dist_version("mlx"),
        "mlx_lm_version": _dist_version("mlx-lm"),
        "jevmlx_version": _dist_version("jevmlx"),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _fmt(value: object) -> str:
    """Markdown cell formatting: None -> n/a, floats to 4 decimals."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: list[str], rows: list[tuple]) -> str:
    """Render a pipe-table; empty rows keep just the header block."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(_fmt(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _accuracy(value: object) -> float:
    """Accuracy of a per_field row, -inf when absent (sorts last, ascending)."""
    return value if isinstance(value, (int, float)) else float("-inf")


def _metric_rows(run: dict) -> list[tuple]:
    """Metrics-table rows: overall accuracy, per-type, then generic metrics.

    Scalar metrics become one row; dicts of scalars become ``name[key]`` rows
    (e.g. majority_class_baseline per field); list-valued or nested metrics
    (risk_coverage curve) stay JSON-only to keep the table readable.
    """
    metrics = run.get("metrics") or {}
    rows: list[tuple] = []
    if "accuracy" in metrics:
        rows.append(("accuracy", metrics["accuracy"]))
    for type_name in sorted(run.get("per_type") or {}):
        entry = run["per_type"][type_name]
        accuracy = entry.get("accuracy") if isinstance(entry, dict) else entry
        rows.append((f"accuracy[{type_name}]", accuracy))
    ordered = (
        "case_exact_match",
        "multi_jaccard",
        "brier",
        "log_loss",
        "correctness_auroc",
        "ece_5bin_equal_mass",
        "tie_rate",
    )
    for key in ordered:
        if key in metrics:
            rows.append((key, metrics[key]))
    handled = set(ordered) | {"accuracy", "risk_coverage"}
    for key in sorted(set(metrics) - handled):
        value = metrics[key]
        if isinstance(value, dict):
            for sub_key, sub_value in sorted(value.items()):
                if isinstance(sub_value, (int, float)):
                    rows.append((f"{key}[{sub_key}]", sub_value))
        elif isinstance(value, (int, float)):
            rows.append((key, value))
    for key in ("latency_ms_p50", "latency_ms_p90"):
        if key in metrics:
            rows.append((key, metrics[key]))
    return rows


def _to_markdown(run: dict) -> str:
    """Markdown summary: environment, metrics, per-field accuracy (ascending)."""
    environment_info = run.get("environment") or {}
    per_field = sorted(
        run.get("per_field") or [],
        key=lambda row: (_accuracy(row.get("accuracy")), str(row.get("field", ""))),
    )
    parts = [
        "# jevmlx eval report",
        "",
        "## Environment",
        "",
        _table(["key", "value"], sorted(environment_info.items())),
        "",
        "## Metrics",
        "",
        _table(["metric", "value"], _metric_rows(run)),
        "",
        "## Per-field accuracy",
        "",
        _table(
            ["field", "n", "accuracy"],
            [(row.get("field"), row.get("n"), row.get("accuracy")) for row in per_field],
        ),
        "",
    ]
    return "\n".join(parts)


def write_report(path: str | Path, run: dict) -> None:
    """Write ``run`` as JSON at ``path`` plus a ``.md`` summary next to it.

    JSON is indented with sorted keys; unknown run keys pass through to JSON
    and are ignored in the markdown.
    """
    path = Path(path)
    path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n")
    path.with_suffix(".md").write_text(_to_markdown(run))
