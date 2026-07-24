import pytest

from knowledge_assistant.evaluation.metrics import (
    AnswerObservation,
    ExpectedCase,
    abstention_accuracy,
    citation_validity,
    fact_coverage,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
)
from knowledge_assistant.evaluation.runner import _assert_baseline, _calibrate_relevance_threshold

pytestmark = pytest.mark.eval


def _cases() -> list[ExpectedCase]:
    return [
        ExpectedCase("a", True, frozenset({"d1", "d2"}), ("30 days",)),
        ExpectedCase("b", True, frozenset({"d3"}), ("receipt",)),
        ExpectedCase("c", False, frozenset()),
    ]


def test_retrieval_metrics_use_only_answerable_cases() -> None:
    rankings = {"a": ["x", "d2"], "b": ["d3"], "c": ["d1"]}

    assert recall_at_k(_cases(), rankings, 2) == pytest.approx(0.75)
    assert mean_reciprocal_rank(_cases(), rankings) == pytest.approx(0.75)


def test_generation_metrics_and_latency() -> None:
    known = frozenset({"d1", "d2", "d3"})
    observations = {
        "a": AnswerObservation(False, ("d1",), known, "Return within 30 days.", 10),
        "b": AnswerObservation(False, ("d3",), known, "Bring a receipt.", 20),
        "c": AnswerObservation(True, (), known, "I do not know.", 100),
    }

    assert abstention_accuracy(_cases(), observations) == 1.0
    assert citation_validity(list(observations.values())) == 1.0
    assert fact_coverage(_cases(), observations) == 1.0
    assert percentile([10, 20, 100], 50) == 20
    assert percentile([10, 20, 100], 95) == 100


def test_invalid_citation_and_empty_inputs_are_handled_explicitly() -> None:
    observation = AnswerObservation(False, ("unknown",), frozenset({"known"}), "answer", 1)

    assert citation_validity([observation]) == 0.0
    assert citation_validity([]) == 1.0
    assert recall_at_k([], {}, 5) == 0.0
    assert mean_reciprocal_rank([], {}) == 0.0
    assert fact_coverage([], {}) == 0.0
    assert percentile([], 95) == 0.0
    with pytest.raises(ValueError, match="percentile"):
        percentile([1], 0)


def test_regression_thresholds_allow_boundary_and_reject_larger_drop() -> None:
    baseline = {"retrieval": {"hybrid": {"recall_at_5": 0.9, "mrr": 0.8}}}
    _assert_baseline(
        {"retrieval": {"hybrid": {"recall_at_5": 0.85, "mrr": 0.75}}},
        baseline,
    )
    with pytest.raises(SystemExit, match="Recall"):
        _assert_baseline(
            {"retrieval": {"hybrid": {"recall_at_5": 0.849, "mrr": 0.8}}},
            baseline,
        )
    with pytest.raises(SystemExit, match="MRR"):
        _assert_baseline(
            {"retrieval": {"hybrid": {"recall_at_5": 0.9, "mrr": 0.749}}},
            baseline,
        )


def test_regression_thresholds_accept_nested_runner_reports() -> None:
    baseline = {"retrieval": {"metrics": {"hybrid": {"recall_at_5": 1.0, "mrr": 0.95}}}}

    _assert_baseline(
        {"retrieval": {"metrics": {"hybrid": {"recall_at_5": 0.95, "mrr": 0.9}}}},
        baseline,
    )


def test_relevance_calibration_prefers_zero_false_positives() -> None:
    result = _calibrate_relevance_threshold(
        [
            ExpectedCase("answerable", True, frozenset({"policy"})),
            ExpectedCase("irrelevant", False, frozenset()),
        ],
        {
            "answerable": [("policy", 0.031), ("other", 0.02)],
            "irrelevant": [("other", 0.03)],
        },
    )

    assert result["false_positives"] == 0
    assert result["answerable_recall"] == 1.0
