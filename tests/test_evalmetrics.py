"""Hand-computable metrics tests for openjev.evalmetrics. No numpy, no model."""

import math

from openjev.evalmetrics import (
    accuracy_cluster_bootstrap,
    any_flip_rate,
    brier_score,
    case_exact_match,
    compute_metrics,
    correctness_auroc,
    ece_equal_mass,
    field_accuracy,
    load_predictions,
    log_loss,
    macro_by,
    majority_class_baseline,
    mean_tvd_across_permutations,
    multi_jaccard,
    risk_coverage_curve,
    tie_rate,
)

# 4 cases, 6 labelled field-predictions. Every number below is hand-checkable.
P = [
    # case c1: action correct (log_scores -> prob APPROVE 0.5, REJECT 0.5)
    {
        "run_id": "r",
        "case_id": "c1",
        "group_id": "g1",
        "source": "typesafe",
        "workflow": "w1",
        "field": "action",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": "APPROVE",
        "prediction": "APPROVE",
        "valid": True,
        "correct": True,
        "log_scores": {"APPROVE": 0.0, "REJECT": 0.0},
        "confidence": 0.6,
    },
    # case c1: amount wrong
    {
        "run_id": "r",
        "case_id": "c1",
        "group_id": "g1",
        "source": "typesafe",
        "workflow": "w1",
        "field": "amount",
        "type": "boolean",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": True,
        "prediction": False,
        "valid": True,
        "correct": False,
        "log_scores": None,
        "confidence": 0.9,
    },
    # case c2: invalid prediction (counts wrong, not skipped)
    {
        "run_id": "r",
        "case_id": "c2",
        "group_id": "g1",
        "source": "quality-eval",
        "workflow": None,
        "field": "action",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": "APPROVE",
        "prediction": None,
        "valid": False,
        "correct": False,
        "log_scores": None,
        "confidence": None,
        "error": "bad JSON",
    },
    # case c3: action correct, high confidence
    {
        "run_id": "r",
        "case_id": "c3",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w1",
        "field": "action",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": "REJECT",
        "prediction": "REJECT",
        "valid": True,
        "correct": True,
        "log_scores": {"APPROVE": -2.0, "REJECT": 2.0},
        "confidence": 0.95,
    },
    # case c3: multi field jaccard 1/2
    {
        "run_id": "r",
        "case_id": "c3",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w1",
        "field": "tags",
        "type": "multi",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": ["a", "b"],
        "prediction": ["a", "c"],
        "valid": True,
        "correct": False,
        "log_scores": None,
        "confidence": 0.7,
    },
    # case c4: multi field correct (jaccard 1.0)
    {
        "run_id": "r",
        "case_id": "c4",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w2",
        "field": "tags",
        "type": "multi",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": ["x"],
        "prediction": ["x"],
        "valid": True,
        "correct": True,
        "log_scores": None,
        "confidence": 0.8,
    },
    # unlabelled row: excluded from accuracies
    {
        "run_id": "r",
        "case_id": "c4",
        "group_id": "g2",
        "source": "typesafe",
        "workflow": "w2",
        "field": "notes",
        "type": "enum",
        "track": "parallel",
        "model": "m",
        "permutation": "canonical",
        "label": None,
        "prediction": "APPROVE",
        "valid": True,
        "correct": None,
        "log_scores": None,
        "confidence": 0.5,
    },
]


class TestAccuracies:
    def test_field_accuracy(self):
        # 6 labelled rows (notes unlabelled): T,F,F,T,F,T = 3/6
        assert field_accuracy(P) == 0.5

    def test_invalid_prediction_counts_wrong(self):
        # drop the correct c3 action: 2/5 labelled correct
        trimmed = [r for r in P if not (r["case_id"] == "c3" and r["field"] == "action")]
        assert field_accuracy(trimmed) == 2 / 5

    def test_case_exact_match(self):
        # c1 wrong (amount), c2 wrong (invalid), c3 wrong (tags), c4 perfect -> 1/4
        assert case_exact_match(P) == 0.25

    def test_macro_by_source(self):
        macro = macro_by(P, "source")
        # typesafe: c1 action T, c1 amount F, c3 action T, c3 tags F, c4 tags T -> 3/5
        assert macro["typesafe"] == 3 / 5
        # quality-eval: only the invalid c2 row -> 0.0
        assert macro["quality-eval"] == 0.0

    def test_macro_by_workflow_excludes_null(self):
        macro = macro_by(P, "workflow")
        assert set(macro) == {"w1", "w2"}

    def test_majority_class_baseline(self):
        majority = majority_class_baseline(P)
        # action labels: APPROVE, APPROVE, REJECT -> majority APPROVE = 2/3
        assert majority["action"] == 2 / 3
        # amount labels: [True] -> 1.0
        assert majority["amount"] == 1.0
        # tags labels: ["a","b"], ["x"] -> each modal once = 1/2
        assert majority["tags"] == 0.5


class TestProbabilistic:
    def test_brier_score(self):
        # scored rows: c1 action dist (0.5, 0.5) label APPROVE -> 0.25+0.25 = 0.5
        # c3 action softmax(2,-2): p_REJECT = e^2/(e^2+e^-2) -> miss (1-p)^2 + hit 0^2
        p_reject = math.exp(2) / (math.exp(2) + math.exp(-2))
        expected = (0.5 + 2 * (1 - p_reject) ** 2) / 2  # mean over 2 rows; c3 has both terms
        assert math.isclose(brier_score(P), expected, rel_tol=1e-9)

    def test_log_loss(self):
        # c1: p(APPROVE)=0.5 -> -ln 0.5 ; c3: p(REJECT) ~= 0.982 -> -ln(0.982)
        p_reject = math.exp(2) / (math.exp(2) + math.exp(-2))
        expected = (-math.log(0.5) + -math.log(p_reject)) / 2
        assert math.isclose(log_loss(P), expected, rel_tol=1e-9)

    def test_correctness_auroc(self):
        # confident-correct pairs: (0.6,True) vs negatives (0.9,False), (0.7,False)
        # (0.95,True) beats both negatives; (0.6,True) beats none -> (2+0)/4 = 0.5
        assert correctness_auroc(P) == 0.5

    def test_auroc_needs_both_classes(self):
        assert correctness_auroc([r for r in P if r.get("confidence") and r.get("correct")]) is None

    def test_ece_equal_mass(self):
        # confidences: 0.6,0.9,0.95,0.7,0.8 sorted -> 5 bins of 1
        # |0.6-1| + |0.7-0| + |0.8-1| + |0.9-0| + |0.95-1| = 0.4+0.7+0.2+0.9+0.05 = 2.25 /5
        assert math.isclose(ece_equal_mass(P, bins=5), 2.25 / 5)

    def test_ece_empty_is_none(self):
        assert ece_equal_mass([], bins=5) is None

    def test_tie_rate(self):
        # c1 action log_scores equal -> tie; c3 not. 1 tie / 2 scored = 0.5
        assert tie_rate(P) == 0.5


class TestSetMetrics:
    def test_multi_jaccard(self):
        # {a,c} vs {a,b} = 1/3, and {x} vs {x} = 1.0 -> 2/3
        assert multi_jaccard(P) == 2 / 3

    def test_multi_jaccard_invalid_is_zero(self):
        invalid = [dict(P[4], valid=False, prediction=None)]
        assert multi_jaccard(invalid) == 0.0


class TestPositionBias:
    def test_any_flip_rate(self):
        # c3 action: canonical REJECT + rot1 APPROVE + rot2 REJECT -> 1 flip / 2 perms
        records = P + [
            {
                "run_id": "r",
                "case_id": "c3",
                "group_id": "g2",
                "source": "typesafe",
                "workflow": "w1",
                "field": "action",
                "type": "enum",
                "track": "parallel",
                "model": "m",
                "permutation": "rot1",
                "label": "REJECT",
                "prediction": "APPROVE",
                "valid": True,
                "correct": False,
                "log_scores": None,
                "confidence": 0.9,
            },
            {
                "run_id": "r",
                "case_id": "c3",
                "group_id": "g2",
                "source": "typesafe",
                "workflow": "w1",
                "field": "action",
                "type": "enum",
                "track": "parallel",
                "model": "m",
                "permutation": "rot2",
                "label": "REJECT",
                "prediction": "REJECT",
                "valid": True,
                "correct": True,
                "log_scores": None,
                "confidence": 0.9,
            },
        ]
        flips = any_flip_rate(records)
        assert flips == {"action": 0.5}

    def test_mean_tvd_across_permutations(self):
        # c1 action canonical log_scores {A:0, R:0} -> dist (0.5, 0.5)
        # rot1 {A:0, R:2} -> dist (0.119, 0.881) -> TVD = 0.381
        records = P + [
            {
                "run_id": "r",
                "case_id": "c1",
                "group_id": "g1",
                "source": "typesafe",
                "workflow": "w1",
                "field": "action",
                "type": "enum",
                "track": "parallel",
                "model": "m",
                "permutation": "rot1",
                "label": "APPROVE",
                "prediction": "REJECT",
                "valid": True,
                "correct": False,
                "log_scores": {"APPROVE": 0.0, "REJECT": 2.0},
                "confidence": 0.9,
            },
        ]
        tvds = mean_tvd_across_permutations(records)
        p_a = math.exp(0) / (math.exp(0) + math.exp(2))
        expected = 0.5 * (abs(0.5 - p_a) + abs(0.5 - (1 - p_a)))
        assert math.isclose(tvds["action"], expected, rel_tol=1e-9)


class TestRiskCoverage:
    def test_curve_shape_and_values(self):
        curve = risk_coverage_curve(P, points=10)
        assert len(curve) == 10
        assert curve[0]["coverage"] == 0.1 and curve[-1]["coverage"] == 1.0
        # full-coverage risk = 1 - 0.6 = 0.4 (5 scored rows, 3 correct)
        assert math.isclose(curve[-1]["risk"], 0.4)
        # top-1 by confidence = 0.95 (correct) -> risk 0 at coverage 0.1
        assert curve[0]["risk"] == 0.0

    def test_empty(self):
        assert risk_coverage_curve([]) == []


class TestBootstrap:
    def test_cluster_bootstrap_moves_whole_cases(self):
        result = accuracy_cluster_bootstrap(P, draws=200, seed=7)
        assert result["n_cases"] == 4
        assert result["n_fields"] == 6
        assert result["accuracy"] == 0.5
        assert 0.0 <= result["ci_low"] <= result["accuracy"] <= result["ci_high"] <= 1.0
        # deterministic given the seed
        again = accuracy_cluster_bootstrap(P, draws=200, seed=7)
        assert (again["ci_low"], again["ci_high"]) == (result["ci_low"], result["ci_high"])


class TestAssembly:
    def test_compute_metrics_keys(self):
        metrics = compute_metrics(P)
        expected_keys = {
            "accuracy",
            "case_exact_match",
            "majority_class_baseline",
            "multi_jaccard",
            "brier",
            "log_loss",
            "correctness_auroc",
            "ece_5bin_equal_mass",
            "tie_rate",
            "risk_coverage",
            "accuracy_cluster_bootstrap",
        }
        assert expected_keys <= set(metrics)

    def test_compute_metrics_empty_is_safe(self):
        assert compute_metrics([]) == {}

    def test_load_predictions_roundtrip(self, tmp_path):
        path = tmp_path / "pred.jsonl"
        path.write_text("\n".join(__import__("json").dumps(r) for r in P[:3]) + "\n\n")
        loaded = load_predictions(path)
        assert len(loaded) == 3 and loaded[0]["case_id"] == "c1"


class TestReportWiring:
    def test_openjev_report_cli_writes_both_files(self, tmp_path, capsys, monkeypatch):
        import json

        from openjev import cli

        predictions = tmp_path / "predictions.jsonl"
        predictions.write_text("\n".join(json.dumps(r) for r in P) + "\n", encoding="utf-8")
        out = tmp_path / "report.json"
        monkeypatch.setattr(
            "sys.argv",
            ["openjev", "report", "--predictions", str(predictions), "--out", str(out)],
        )
        cli.main()
        run = json.loads(out.read_text())
        assert run["metrics"]["accuracy"] == 0.5
        md = out.with_suffix(".md").read_text()
        assert "## Metrics" in md and "| accuracy |" in md
