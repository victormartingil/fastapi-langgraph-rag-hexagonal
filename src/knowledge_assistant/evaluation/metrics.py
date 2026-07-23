"""Deterministic evaluation metrics with no model or infrastructure dependency."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class ExpectedCase:
    case_id: str
    answerable: bool
    relevant_documents: frozenset[str]
    expected_facts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerObservation:
    refused: bool
    cited_documents: tuple[str, ...]
    known_documents: frozenset[str]
    answer_text: str
    latency_ms: float


def recall_at_k(
    cases: Sequence[ExpectedCase],
    rankings: Mapping[str, Sequence[str]],
    k: int,
) -> float:
    """Macro-average fraction of relevant documents retrieved in the top k."""
    answerable = [case for case in cases if case.relevant_documents]
    if not answerable:
        return 0.0
    return sum(
        len(case.relevant_documents.intersection(rankings.get(case.case_id, ())[:k]))
        / len(case.relevant_documents)
        for case in answerable
    ) / len(answerable)


def mean_reciprocal_rank(
    cases: Sequence[ExpectedCase], rankings: Mapping[str, Sequence[str]]
) -> float:
    """Mean reciprocal rank of the first relevant result."""
    answerable = [case for case in cases if case.relevant_documents]
    if not answerable:
        return 0.0
    reciprocal_ranks: list[float] = []
    for case in answerable:
        rank = next(
            (
                index
                for index, document_id in enumerate(rankings.get(case.case_id, ()), start=1)
                if document_id in case.relevant_documents
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def abstention_accuracy(
    cases: Sequence[ExpectedCase], observations: Mapping[str, AnswerObservation]
) -> float:
    if not cases:
        return 0.0
    return sum(observations[case.case_id].refused is (not case.answerable) for case in cases) / len(
        cases
    )


def citation_validity(observations: Sequence[AnswerObservation]) -> float:
    """Fraction of citations that resolve to documents in the evaluated corpus."""
    citations = [
        citation for observation in observations for citation in observation.cited_documents
    ]
    if not citations:
        return 1.0
    valid = sum(
        citation in observation.known_documents
        for observation in observations
        for citation in observation.cited_documents
    )
    return valid / len(citations)


def fact_coverage(
    cases: Sequence[ExpectedCase], observations: Mapping[str, AnswerObservation]
) -> float:
    """Macro-average expected fact phrases present in generated answers."""
    scored = [case for case in cases if case.answerable and case.expected_facts]
    if not scored:
        return 0.0
    return sum(
        sum(
            fact.casefold() in observations[case.case_id].answer_text.casefold()
            for fact in case.expected_facts
        )
        / len(case.expected_facts)
        for case in scored
    ) / len(scored)


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Nearest-rank percentile, deterministic for small benchmark samples."""
    if not values:
        return 0.0
    if not 0 < percentile_value <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    return ordered[max(0, ceil(percentile_value / 100 * len(ordered)) - 1)]
