"""Eval metrics over predictions.jsonl (see E-contracts).

Pure Python + math, no third-party dependencies. Every metric treats an
invalid prediction as wrong (unconditional field accuracy); ``correct`` is
respected when present (null label -> excluded from accuracy denominators).
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

__all__ = ["load_predictions", "compute_metrics"]


def load_predictions(path: str | Path) -> list[dict]:
    """Read a predictions.jsonl file (one JSON object per non-blank line)."""
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _valid_correct(record: dict) -> bool:
    """True iff this prediction counts as correct (invalid => wrong)."""
    if record.get("correct") is not None:
        return bool(record["correct"])
    return bool(record.get("valid")) and record.get("prediction") is not None


def _labelled(records: list[dict]) -> list[dict]:
    """Records that have a label (correct may be False when invalid)."""
    return [r for r in records if r.get("label") is not None or r.get("correct") is not None]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _tvd(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Total variation distance between two distributions over a shared support."""
    keys = set(dist_a) | set(dist_b)
    return 0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


# ---------------------------------------------------------------- accuracies


def field_accuracy(records: list[dict]) -> float | None:
    """Unconditional field accuracy: correct / labelled, invalid counts wrong."""
    labelled = _labelled(records)
    if not labelled:
        return None
    return sum(1 for r in labelled if _valid_correct(r)) / len(labelled)


def case_exact_match(records: list[dict]) -> float | None:
    """Share of cases where every labelled field is correct (invalid = wrong)."""
    by_case: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_case[record["case_id"]].append(record)
    if not by_case:
        return None
    perfect = 0
    for case_records in by_case.values():
        labelled = _labelled(case_records)
        if labelled and all(_valid_correct(r) for r in labelled):
            perfect += 1
    return perfect / len(by_case)


def macro_by(records: list[dict], key: str) -> dict[str, float]:
    """Macro (per-group unweighted mean) field accuracy grouped by ``key``.

    Records whose ``key`` is None are skipped (a null workflow is no group).
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record.get(key) is None:
            continue
        grouped[str(record[key])].append(record)
    macro = {}
    for group, group_records in grouped.items():
        accuracy = field_accuracy(group_records)
        if accuracy is not None:
            macro[group] = accuracy
    return dict(sorted(macro.items()))


def majority_class_baseline(records: list[dict]) -> dict[str, float]:
    """Per-field accuracy of always predicting the modal labelled value."""
    by_field: dict[str, list[dict]] = defaultdict(list)
    for record in _labelled(records):
        by_field[record["field"]].append(record)
    baseline: dict[str, float] = {}
    for field, field_records in by_field.items():
        counts = Counter(
            json.dumps(r["label"], sort_keys=True)
            if isinstance(r["label"], (list, dict))
            else r["label"]
            for r in field_records
        )
        baseline[field] = counts.most_common(1)[0][1] / len(field_records)
    return dict(sorted(baseline.items()))


# ------------------------------------------------------------- probabilistic


def brier_score(records: list[dict]) -> float | None:
    """Multiclass Brier: mean over fields of sum over choices of (p - y)^2."""
    scores = []
    for record in records:
        distribution = _record_distribution(record)
        if distribution is None:
            continue
        label = record.get("label")
        scores.append(
            sum(
                (prob - (1.0 if choice == label else 0.0)) ** 2
                for choice, prob in distribution.items()
            )
        )
    return _mean(scores)


def log_loss(records: list[dict]) -> float | None:
    """Multiclass true-label log loss (natural log)."""
    losses = []
    for record in records:
        distribution = _record_distribution(record)
        if distribution is None:
            continue
        label = record.get("label")
        prob = distribution.get(label)
        if prob is None or not _finite(prob) or prob <= 0:
            return None  # unmeasurable distribution; never paper over it
        losses.append(-math.log(prob))
    return _mean(losses)


def _record_distribution(record: dict) -> dict[str, float] | None:
    """Best available choice distribution: log_scores softmaxed at T=1."""
    log_scores = record.get("log_scores")
    if not isinstance(log_scores, dict) or not log_scores:
        return None
    if not all(_finite(v) for v in log_scores.values()):
        return None
    peak = max(log_scores.values())
    exp = {choice: math.exp(score - peak) for choice, score in log_scores.items()}
    total = sum(exp.values())
    if total <= 0:
        return None
    return {choice: value / total for choice, value in exp.items()}


def correctness_auroc(records: list[dict]) -> float | None:
    """AUROC of confidence predicting correctness (rank-statistic, no numpy).

    Labelled records only; pairs with equal confidence count 0.5. Returns
    None without both classes.
    """
    scored = [
        (float(r["confidence"]), _valid_correct(r))
        for r in _labelled(records)
        if r.get("confidence") is not None and _finite(r["confidence"])
    ]
    positive = sum(1 for _, correct in scored if correct)
    negative = len(scored) - positive
    if not positive or not negative:
        return None
    wins = 0.0
    for confidence_a, correct_a in scored:
        for confidence_b, correct_b in scored:
            if not correct_a or correct_b:
                continue  # only positive-vs-negative pairs contribute
            if confidence_a > confidence_b:
                wins += 1.0
            elif confidence_a == confidence_b:
                wins += 0.5
    return wins / (positive * negative)


def ece_equal_mass(records: list[dict], bins: int = 5) -> float | None:
    """Descriptive ECE over ``bins`` equal-mass confidence bins.

    Empty-sample safety: returns None when there is nothing to measure.
    """
    scored = [
        (float(r["confidence"]), 1.0 if _valid_correct(r) else 0.0)
        for r in _labelled(records)
        if r.get("confidence") is not None and _finite(r["confidence"])
    ]
    if not scored:
        return None
    scored.sort()
    total = len(scored)
    bin_size = math.ceil(total / bins)
    ece = 0.0
    for start in range(0, total, bin_size):
        chunk = scored[start : start + bin_size]
        mean_confidence = sum(c for c, _ in chunk) / len(chunk)
        mean_correct = sum(y for _, y in chunk) / len(chunk)
        ece += (len(chunk) / total) * abs(mean_confidence - mean_correct)
    return ece


# ------------------------------------------------------------------ set fields


def multi_jaccard(records: list[dict]) -> float | None:
    """Mean Jaccard similarity of multi predictions (invalid => 0.0)."""
    scores = []
    for record in records:
        if record.get("type") != "multi":
            continue
        if record.get("label") is None:
            continue
        predicted = record.get("prediction")
        if not record.get("valid") or not isinstance(predicted, list):
            scores.append(0.0)
            continue
        predicted_set, label_set = set(predicted), set(record["label"])
        union = predicted_set | label_set
        scores.append(len(predicted_set & label_set) / len(union) if union else 1.0)
    return _mean(scores)


# -------------------------------------------------------- position-bias (permutation)


def any_flip_rate(records: list[dict]) -> dict[str, float]:
    """Per field: share of permuted runs whose chosen value differs from canonical."""
    canonical: dict[tuple[str, str], object] = {}
    permuted: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = (record["case_id"], record["field"])
        if (record.get("permutation") or "canonical") == "canonical":
            canonical[key] = record.get("prediction")
        else:
            permuted[record["field"]].append(record)
    flips: dict[str, float] = {}
    for field, field_records in permuted.items():
        comparable = [r for r in field_records if (r["case_id"], field) in canonical]
        if not comparable:
            continue
        flipped = sum(
            1
            for r in comparable
            if json.dumps(r.get("prediction"), sort_keys=True)
            != json.dumps(canonical[(r["case_id"], field)], sort_keys=True)
        )
        flips[field] = flipped / len(comparable)
    return dict(sorted(flips.items()))


def mean_tvd_across_permutations(records: list[dict]) -> dict[str, float]:
    """Per field: mean TVD between each permutation's distribution and canonical's.

    Uses the softmaxed log_scores distribution; requires a canonical record
    per (case, field). Skips permutations lacking one.
    """
    canonical: dict[tuple[str, str], dict] = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("log_scores") is None:
            continue
        key = (record["case_id"], record["field"])
        if record.get("permutation") == "canonical":
            canonical[key] = record
        else:
            grouped[key].append(record)
    tvds: dict[str, list[float]] = defaultdict(list)
    for key, permuted_records in grouped.items():
        base = canonical.get(key)
        if base is None:
            continue
        base_distribution = _record_distribution(base)
        if base_distribution is None:
            continue
        for record in permuted_records:
            distribution = _record_distribution(record)
            if distribution is None:
                continue
            tvds[record["field"]].append(_tvd(base_distribution, distribution))
    return {field: _mean(values) for field, values in sorted(tvds.items()) if values}


def tie_rate(records: list[dict]) -> float | None:
    """Share of scored records whose top-two log_scores are exactly equal."""
    scored = [r for r in records if isinstance(r.get("log_scores"), dict) and r["log_scores"]]
    if not scored:
        return None
    ties = 0
    for record in scored:
        values = sorted(record["log_scores"].values(), reverse=True)
        if len(values) >= 2 and values[0] == values[1]:
            ties += 1
    return ties / len(scored)


# -------------------------------------------------------------- risk-coverage


def risk_coverage_curve(records: list[dict], points: int = 10) -> list[dict]:
    """Risk (1 - accuracy) at ``points`` evenly spaced coverage levels.

    Records are sorted by confidence descending (tie-break: stable order);
    at each coverage the prefix risk is computed. Uncovered records count as
    neither right nor wrong — they are simply not yet included.
    """
    scored = sorted(
        (
            (float(r["confidence"]), 1.0 if _valid_correct(r) else 0.0)
            for r in _labelled(records)
            if r.get("confidence") is not None and _finite(r["confidence"])
        ),
        key=lambda pair: -pair[0],
    )
    total = len(scored)
    if total == 0:
        return []
    curve = []
    for i in range(1, points + 1):
        coverage = i / points
        take = max(1, min(total, math.ceil(coverage * total)))
        prefix = scored[:take]
        accuracy = sum(y for _, y in prefix) / len(prefix)
        curve.append({"coverage": round(coverage, 2), "risk": 1.0 - accuracy, "n": take})
    return curve


# ------------------------------------------------------------------ bootstrap


def accuracy_cluster_bootstrap(
    records: list[dict],
    draws: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict | None:
    """Case-level cluster bootstrap CI for field accuracy.

    Resamples whole cases (all their fields move together) ``draws`` times and
    returns point estimate plus the central percentile interval.
    """
    by_case: dict[str, list[dict]] = defaultdict(list)
    for record in _labelled(records):
        by_case[record["case_id"]].append(record)
    case_ids = sorted(by_case)
    if not case_ids:
        return None
    point = field_accuracy(records)
    rng = random.Random(seed)
    stats = []
    for _ in range(draws):
        sample = [r for case in rng.choices(case_ids, k=len(case_ids)) for r in by_case[case]]
        stats.append(field_accuracy(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2
    lower_index = math.floor(alpha * (draws - 1))
    upper_index = math.ceil((1 - alpha) * (draws - 1))
    return {
        "accuracy": point,
        "ci_low": stats[lower_index],
        "ci_high": stats[upper_index],
        "n_cases": len(case_ids),
        "n_fields": sum(len(v) for v in by_case.values()),
        "draws": draws,
        "seed": seed,
    }


# ------------------------------------------------------------------- assembly


def compute_metrics(records: list[dict]) -> dict:
    """Assemble the metrics dict for evalreport.write_report.

    Missing-data safety: a metric that cannot be computed (no labelled rows,
    no confidences, no log_scores) is simply absent rather than zero-filled.
    """
    metrics: dict = {}
    accuracy = field_accuracy(records)
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    exact = case_exact_match(records)
    if exact is not None:
        metrics["case_exact_match"] = exact
    majority = majority_class_baseline(records)
    if majority:
        metrics["majority_class_baseline"] = majority
    jaccard = multi_jaccard(records)
    if jaccard is not None:
        metrics["multi_jaccard"] = jaccard
    brier = brier_score(records)
    if brier is not None:
        metrics["brier"] = brier
    nll = log_loss(records)
    if nll is not None:
        metrics["log_loss"] = nll
    auroc = correctness_auroc(records)
    if auroc is not None:
        metrics["correctness_auroc"] = auroc
    ece = ece_equal_mass(records, bins=5)
    if ece is not None:
        metrics["ece_5bin_equal_mass"] = ece
    ties = tie_rate(records)
    if ties is not None:
        metrics["tie_rate"] = ties
    curve = risk_coverage_curve(records, points=10)
    if curve:
        metrics["risk_coverage"] = curve
    flips = any_flip_rate(records)
    if flips:
        metrics["any_flip_rate"] = flips
    tvds = mean_tvd_across_permutations(records)
    if tvds:
        metrics["mean_tvd"] = tvds
    bootstrap = accuracy_cluster_bootstrap(records)
    if bootstrap:
        metrics["accuracy_cluster_bootstrap"] = bootstrap
    return metrics
