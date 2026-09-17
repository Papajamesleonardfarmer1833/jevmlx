"""Temperature calibration for the parallel decision engine.

One scalar T fitted on labeled cases minimizes the mean NLL of the true
choice under softmax(scores / T). No scipy, no numpy — stdlib math only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

Sample = tuple[list[float], int]  # (per-choice scores at T=1, labeled choice index)


def collect(model, tokenizer, cases: Sequence[dict]) -> list[Sample]:
    """Run run_parallel_generation once per case at T=1 and keep raw scores
    for every labeled field. Cases: [{"schema": {...}, "context": str,
    "labels": {field: value}}] (labels may cover a subset of fields).

    multi fields are skipped with a logged warning: their scores are
    per-option p_true values, not a choice distribution, so NLL fitting
    does not apply to them."""
    import logging

    from openjev.engine import run_parallel_generation
    from openjev.schema import StructuredSchema

    logger = logging.getLogger(__name__)

    samples: list[Sample] = []
    for case in cases:
        schema = StructuredSchema(case["schema"])
        result = run_parallel_generation(model, tokenizer, case["context"], schema, temperature=1.0)
        for fname, label in case["labels"].items():
            telemetry = result["field_telemetry"][fname]
            fdef = schema[fname]
            if fdef.field_type == "multi":
                logger.warning("calibrate.collect: skipping multi field '%s'", fname)
                continue
            choices = ["true", "false"] if fdef.field_type == "boolean" else fdef.choices
            label_str = (
                str(label) if fdef.field_type != "boolean" else ("true" if label else "false")
            )
            if label_str not in choices:
                raise ValueError(f"case label {fname}={label!r} not in choices {choices}")
            log_scores = telemetry["log_scores"]
            samples.append(([log_scores[choice] for choice in choices], choices.index(label_str)))
    return samples


def _nll(scores: Sequence[float], label_idx: int, t: float) -> float:
    scaled = [s / t for s in scores]
    m = max(scaled)
    log_z = m + math.log(sum(math.exp(s - m) for s in scaled))
    return log_z - scaled[label_idx]


def fit_temperature(
    samples: Sequence[Sample], lo: float = 0.05, hi: float = 20.0, iters: int = 60
) -> float:
    """Golden-section search for the T in [lo, hi] minimizing mean NLL."""

    def mean_nll(t: float) -> float:
        return sum(_nll(s, y, t) for s, y in samples) / len(samples)

    inv_phi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = mean_nll(c), mean_nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = mean_nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = mean_nll(d)
    return round((a + b) / 2, 4)


def _probs(scores: Sequence[float], t: float) -> list[float]:
    scaled = [s / t for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def ece(samples: Sequence[Sample], t: float = 1.0, bins: int = 10) -> float:
    """Expected calibration error: mean |max-prob - accuracy| over equal-width
    confidence bins, weighted by bin size."""
    if not samples:
        return 0.0
    bin_conf = [0.0] * bins
    bin_acc = [0.0] * bins
    bin_n = [0] * bins
    for scores, label_idx in samples:
        probs = _probs(scores, t)
        p_max = max(probs)
        correct = 1.0 if max(range(len(probs)), key=probs.__getitem__) == label_idx else 0.0
        b = min(bins - 1, int(p_max * bins))
        bin_conf[b] += p_max
        bin_acc[b] += correct
        bin_n[b] += 1
    total = len(samples)
    return sum(
        (bin_n[i] / total) * abs(bin_conf[i] / bin_n[i] - bin_acc[i] / bin_n[i])
        for i in range(bins)
        if bin_n[i]
    )


def load_cases(path: str) -> list:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def accuracy(samples: Sequence[Sample]) -> float:
    """Fraction of samples whose argmax choice matches the label."""
    if not samples:
        return 0.0
    hits = sum(1 for s, y in samples if max(range(len(s)), key=lambda i: s[i]) == y)
    return hits / len(samples)
