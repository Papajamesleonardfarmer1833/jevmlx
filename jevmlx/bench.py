"""``jevmlx bench``: one command, a complete PR-ready benchmark results folder.

Runs the eval harness across tracks x scorers x bundled datasets on this
machine, writes predictions/run manifests/reports per combination, then
summarizes everything into ``SUMMARY.md`` with PR instructions. Datasets are
built once into ``~/.cache/jevmlx/bench/`` and reused while their lock files
match.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from jevmlx.engine import clear_engine_cache, load_engine
from jevmlx.evalmetrics import compute_metrics, load_predictions
from jevmlx.evalreport import environment, write_report
from jevmlx.evalrun import parallel_decide_fn, run_eval

BENCH_CACHE = Path.home() / ".cache" / "jevmlx" / "bench"
HERE = Path(__file__).resolve().parent.parent / "benchmarks"
RESULTS_DIR = HERE / "results"
MAX_FOLDER_BYTES = 5 * 1024 * 1024

DATASETS = ("bundled", "typesafe", "perturbed")
SCORERS = ("trie", "letters")
TRACKS = ("parallel", "naive_local")


def machine_tag(override: str | None = None) -> str:
    """Machine tag ``<chip-lowercase>-<ram>gb`` (e.g. ``m5max-128gb``).

    The chip comes from ``sysctl machdep.cpu.brand_string`` with the marketing
    noise dropped (Apple M5 Max -> m5max); RAM from ``hw.memsize`` in GB.
    """
    if override:
        return override
    chip_raw = _sysctl(["machdep.cpu.brand_string"]) or "unknown-chip"
    words = [w for w in chip_raw.replace("Apple", "").split() if w]
    chip = "".join(w.lower() for w in words if w.lower() != "apple")
    ram_raw = _sysctl(["hw.memsize"])
    ram_gb = int(int(ram_raw) / 2**30) if (ram_raw or "").isdigit() else 0
    return f"{chip}-{ram_gb}gb"


def _sysctl(name_args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", *name_args], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def preflight(force: bool, machine_override: str | None) -> str:
    """Refuse to run on machines that cannot produce comparable numbers.

    Checks Apple Silicon, free Metal memory (a second GPU-heavy process would
    skew latency), and battery power (thermal throttling). ``--force``
    overrides the battery and Metal checks (never the platform check) with
    the reason printed. Returns the machine tag.
    """
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("bench requires Apple Silicon (macOS + arm64).")

    tag = machine_tag(machine_override)

    battery = _probe(["pmset", "-g", "batt"])
    on_battery = battery is not None and "AC Power" not in battery
    if on_battery:
        if not force:
            raise SystemExit(
                "on battery power (thermal throttling skews latency); plug in or pass --force"
            )
        print("WARNING: on battery power (--force); latency numbers may be throttled.")

    if not force:
        resident = _metal_resident_bytes()
        if resident is not None and resident > 1 * 2**30:
            raise SystemExit(
                f"another process holds ~{resident / 2**30:.1f} GB of Metal memory; "
                "close it or pass --force"
            )
    return tag


def _probe(command: list[str]) -> str | None:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _metal_resident_bytes() -> int | None:
    """Wired memory in bytes as a proxy for Metal pressure (other GPU processes
    would show up as wired/in-use memory)."""
    raw = _sysctl(["vm.wire_count"])
    if raw is None or not raw.isdigit():
        return None
    page = _sysctl(["hw.pagesize"])
    page_size = int(page) if (page or "").isdigit() else 16384
    return int(raw) * page_size


def model_slug(model_id: str) -> str:
    """Folder-safe model slug: lowercased, '/' -> '--', dots kept."""
    return model_id.lower().replace("/", "--")


def build_datasets(datasets: list[str], offline_ok: bool = True) -> dict[str, Path]:
    """Build eval JSONL datasets into the bench cache, reusing lock matches.

    Returns dataset name -> JSONL path. ``typesafe`` is skipped with a clear
    message when offline; ``perturbed`` derives from ``bundled``.
    """
    BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    if "bundled" in datasets:
        jsonl = BENCH_CACHE / "bundled.jsonl"
        lock = BENCH_CACHE / "bundled.dataset.lock.json"
        _rebuild_if_needed(jsonl, lock, _build_bundled)
        paths["bundled"] = jsonl

    if "typesafe" in datasets:
        jsonl = BENCH_CACHE / "typesafe.jsonl"
        lock = BENCH_CACHE / "typesafe.dataset.lock.json"
        try:
            _rebuild_if_needed(jsonl, lock, _build_typesafe)
        except OSError as exc:
            if offline_ok and not (jsonl.exists() and lock.exists()):
                print(f"typesafe dataset skipped (offline): {exc}")
            elif jsonl.exists() and lock.exists():
                print(f"typesafe dataset: reusing cached copy (fetch failed: {exc})")
            else:
                raise
        paths["typesafe"] = jsonl

    if "perturbed" in datasets:
        jsonl = BENCH_CACHE / "perturbed.jsonl"
        lock = BENCH_CACHE / "perturbed.dataset.lock.json"
        _rebuild_if_needed(jsonl, lock, lambda: _build_perturbed(paths["bundled"]))
        paths["perturbed"] = jsonl

    return paths


def _rebuild_if_needed(jsonl: Path, lock: Path, build) -> None:
    """Rebuild the dataset when missing or when the previous build was partial."""
    if jsonl.exists() and lock.exists():
        return
    build()


def _build_bundled() -> None:
    from benchmarks.to_jsonl import main as to_jsonl_main

    out = str(BENCH_CACHE / "bundled.jsonl")
    print("building bundled dataset...")
    to_jsonl_main(["--out", out])


def _build_typesafe() -> None:
    from benchmarks.typesafe.fetch import main as fetch_main

    out = str(BENCH_CACHE / "typesafe.jsonl")
    print("building typesafe dataset (downloads from the network)...")
    rc = fetch_main(["--out", out])
    if rc != 0:
        raise OSError("typesafe fetch failed")


def _build_perturbed(bundled: Path) -> None:
    from benchmarks.perturb import main as perturb_main

    print("building perturbed dataset from bundled...")
    perturb_main(
        [
            "--in",
            str(bundled),
            "--out",
            str(BENCH_CACHE / "perturbed.jsonl"),
            "--variants",
            "3",
            "--seed",
            "0",
        ]
    )


def _track_scorer_grid(tracks: list[str], scorers: list[str]) -> list[tuple[str, str]]:
    """Valid (track, scorer) pairs: naive_local is scorer-independent."""
    grid = []
    for track in tracks:
        for scorer in scorers:
            if track == "naive_local" and scorer != "trie":
                continue  # naive generation has no scorer dimension
            grid.append((track, scorer))
    return grid


def run_bench(
    model: str,
    datasets: list[str],
    scorers: list[str],
    tracks: list[str],
    out: Path,
    runs: int,
    machine_override: str | None = None,
    force: bool = False,
) -> Path:
    """Run the full bench matrix; returns the results folder path."""
    tag = preflight(force, machine_override)
    print(f"machine: {tag}")

    dataset_paths = build_datasets(datasets)
    if not dataset_paths:
        raise SystemExit("no datasets selected")

    slug = model_slug(model)
    folder = out / f"{tag}-{slug}"
    folder.mkdir(parents=True, exist_ok=True)

    # Dataset locks travel with the results.
    for name in dataset_paths:
        lock_src = BENCH_CACHE / f"{name}.dataset.lock.json"
        if lock_src.exists():
            shutil.copy(lock_src, folder / f"{name}.dataset.lock.json")

    last_run: dict[str, dict[str, Any]] = {}
    for track, scorer in _track_scorer_grid(tracks, scorers):
        for dataset in datasets:
            if dataset not in dataset_paths:
                continue
            combo = f"{track}-{scorer}-{dataset}"
            combo_dir = folder / combo
            combo_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== {combo} ({runs} run(s)) ===")
            result = None
            for run_index in range(runs):
                result = _run_one(model, track, scorer, dataset_paths[dataset], combo_dir)
                print(f"  run {run_index + 1}/{runs} done")
            assert result is not None
            last_run[combo] = result

        # One model load per track group is enough; drop it between tracks so
        # memory returns to baseline before the next track's runs.
        clear_engine_cache()

    summarize(folder)
    _print_pr_instructions(folder, last_run)
    return folder


def _run_one(model: str, track: str, scorer: str, jsonl: Path, combo_dir: Path) -> dict:
    """One eval run (in-process) + metrics + report, into combo_dir."""
    cases = _load_cases(jsonl)
    model_obj, tokenizer = load_engine(model)
    chat_template = getattr(tokenizer, "chat_template", None)

    if track == "parallel":
        decide_fn = parallel_decide_fn(model_obj, tokenizer, scoring=scorer)
    else:
        from jevmlx.evalrun import naive_local_decide_fn

        decide_fn = naive_local_decide_fn(model_obj, tokenizer)

    permutations = "rotations" if track == "parallel" else "none"
    run = run_eval(
        cases,
        decide_fn,
        track=track,
        model=model,
        permutations=permutations,
        split="all",
        out_dir=str(combo_dir),
        extra_config={"scoring": scorer if track == "parallel" else "trie"},
        chat_template=chat_template,
        dataset_path=str(jsonl),
    )

    records = load_predictions(combo_dir / "predictions.jsonl")
    report_path = combo_dir / "report.json"
    write_report(report_path, {"environment": environment(), "metrics": compute_metrics(records)})
    print(f"  wrote {combo_dir}/predictions.jsonl, run.json, report.json, report.md")
    return {"run": run, "report": report_path}


def _load_cases(jsonl: Path) -> list[dict]:
    cases: list[dict] = []
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                cases.append(__import__("json").loads(line))
    return cases


def summarize(out: Path) -> Path:
    """Summarize every report.json under ``out`` into ``out/SUMMARY.md``."""
    from benchmarks.summarize_results import summarize as _summarize

    return _summarize(out)


def _print_pr_instructions(folder: Path, last_run: dict) -> None:
    n_combos = len(last_run)
    print(
        "\nBench complete.\n"
        f"Results folder: {folder}\n"
        f"Combos: {n_combos}; summary: {folder / 'SUMMARY.md'}\n\n"
        "To publish these numbers:\n"
        f"  1. git checkout -b bench-results-{folder.name}\n"
        f"  2. git add {folder}  (predictions may be gzipped; see the folder README)\n"
        f"  3. git commit -m 'Bench results: {folder.name}'\n"
        "  4. Open a PR against main with SUMMARY.md pasted into the description.\n"
        "Do NOT commit anything outside the results folder (no caches, no models)."
    )


def enforce_folder_size(folder: Path) -> bool:
    """Gzip any predictions.jsonl pushing the folder over 5 MB.

    Returns True when any file was compressed (the report should say so).
    """
    total = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
    if total <= MAX_FOLDER_BYTES:
        return False
    compressed = False
    for pred in folder.rglob("predictions.jsonl"):
        gz_path = pred.with_suffix(".jsonl.gz")
        with open(pred, "rb") as src, gzip.open(gz_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        pred.unlink()
        compressed = True
    return compressed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevmlx bench",
        description="One command: complete PR-ready benchmark results folder.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model id for mlx-lm")
    parser.add_argument(
        "--datasets",
        default="bundled,typesafe,perturbed",
        help="comma list: bundled,typesafe,perturbed",
    )
    parser.add_argument("--scorers", default="trie,letters", help="comma list: trie,letters")
    parser.add_argument(
        "--tracks", default="parallel,naive_local", help="comma list: parallel,naive_local"
    )
    parser.add_argument(
        "--out", default=str(RESULTS_DIR), help="results root (default benchmarks/results)"
    )
    parser.add_argument("--runs", type=int, default=2, help="eval runs per combo (last one kept)")
    parser.add_argument("--machine", default=None, help="override the machine tag")
    parser.add_argument(
        "--force",
        action="store_true",
        help="run despite battery power or busy Metal memory (reasons are printed)",
    )
    args = parser.parse_args(argv)

    datasets = [d for d in (s.strip() for s in args.datasets.split(",")) if d]
    scorers = [s for s in (s.strip() for s in args.scorers.split(",")) if s]
    tracks = [t for t in (t.strip() for t in args.tracks.split(",")) if t]
    for name, values, allowed in (
        ("datasets", datasets, DATASETS),
        ("scorers", scorers, SCORERS),
        ("tracks", tracks, TRACKS),
    ):
        bad = [v for v in values if v not in allowed]
        if bad:
            parser.error(f"invalid {name}: {', '.join(bad)} (allowed: {', '.join(allowed)})")
    if args.runs < 1:
        parser.error("--runs must be >= 1")

    run_bench(
        model=args.model,
        datasets=datasets,
        scorers=scorers,
        tracks=tracks,
        out=Path(args.out),
        runs=args.runs,
        machine_override=args.machine,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
