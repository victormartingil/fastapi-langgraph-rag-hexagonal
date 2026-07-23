"""CLI benchmark for lexical, live dense, hybrid, and end-to-end API quality."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

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

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text)}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def lexical_rank(question: str, documents: Sequence[Mapping[str, Any]]) -> list[str]:
    query_tokens = _tokens(question)
    scored = [
        (
            len(query_tokens.intersection(_tokens(str(document["text"])))),
            str(document["id"]),
        )
        for document in documents
    ]
    return [
        document_id
        for score, document_id in sorted(scored, key=lambda item: (-item[0], item[1]))
        if score > 0
    ]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (rrf_k + rank)
    return [
        document_id
        for document_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]


async def _ollama_embeddings(
    client: httpx.AsyncClient, texts: Sequence[str], model: str
) -> list[list[float]]:
    response = await client.post("/api/embed", json={"model": model, "input": list(texts)})
    response.raise_for_status()
    payload = response.json()
    return [[float(value) for value in vector] for vector in payload["embeddings"]]


def _load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    return payload["documents"], payload["cases"]


def _expected(cases: Sequence[Mapping[str, Any]]) -> list[ExpectedCase]:
    return [
        ExpectedCase(
            case_id=str(case["id"]),
            answerable=bool(case["answerable"]),
            relevant_documents=frozenset(case["relevant_documents"]),
            expected_facts=tuple(case.get("expected_facts", ())),
        )
        for case in cases
    ]


def _retrieval_metrics(
    cases: Sequence[ExpectedCase], rankings: Mapping[str, Sequence[str]]
) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(cases, rankings, 5),
        "mrr": mean_reciprocal_rank(cases, rankings),
    }


async def _evaluate_retrieval(
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    ollama_url: str | None,
    embedding_model: str,
) -> dict[str, dict[str, float]]:
    lexical = {str(case["id"]): lexical_rank(str(case["question"]), documents) for case in cases}
    result = {"lexical": _retrieval_metrics(_expected(cases), lexical)}
    if ollama_url is None:
        return result

    async with httpx.AsyncClient(base_url=ollama_url, timeout=120) as client:
        document_vectors = await _ollama_embeddings(
            client, [str(document["text"]) for document in documents], embedding_model
        )
        question_vectors = await _ollama_embeddings(
            client, [str(case["question"]) for case in cases], embedding_model
        )
    dense: dict[str, list[str]] = {}
    hybrid: dict[str, list[str]] = {}
    for case, query_vector in zip(cases, question_vectors, strict=True):
        ranked = sorted(
            zip(documents, document_vectors, strict=True),
            key=lambda pair: (
                -_cosine(query_vector, pair[1]),
                str(pair[0]["id"]),
            ),
        )
        case_id = str(case["id"])
        dense[case_id] = [str(document["id"]) for document, _ in ranked]
        hybrid[case_id] = reciprocal_rank_fusion([dense[case_id], lexical[case_id]])
    expected = _expected(cases)
    result["dense"] = _retrieval_metrics(expected, dense)
    result["hybrid"] = _retrieval_metrics(expected, hybrid)
    return result


async def _evaluate_api(
    api_url: str,
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, float]:
    title_to_id = {str(document["title"]): str(document["id"]) for document in documents}
    known_documents = frozenset(title_to_id.values())
    observations: dict[str, AnswerObservation] = {}
    async with httpx.AsyncClient(base_url=api_url, timeout=180) as client:
        for document in documents:
            response = await client.post(
                "/api/v1/documents",
                files={
                    "file": (
                        f"{document['id']}.txt",
                        str(document["text"]).encode(),
                        "text/plain",
                    )
                },
                data={"title": str(document["title"])},
            )
            if response.status_code not in {200, 201}:
                response.raise_for_status()
        for case in cases:
            started = time.perf_counter()
            response = await client.post(
                "/api/v1/chat",
                json={"question": str(case["question"]), "top_k": 5},
            )
            response.raise_for_status()
            elapsed_ms = (time.perf_counter() - started) * 1_000
            payload = response.json()
            cited = tuple(
                title_to_id.get(str(source["document_title"]), "<unknown>")
                for source in payload["sources"]
            )
            observations[str(case["id"])] = AnswerObservation(
                refused=not cited,
                cited_documents=cited,
                known_documents=known_documents,
                answer_text=str(payload["answer"]),
                latency_ms=elapsed_ms,
            )
    expected = _expected(cases)
    values = list(observations.values())
    return {
        "abstention_accuracy": abstention_accuracy(expected, observations),
        "citation_validity": citation_validity(values),
        "fact_coverage": fact_coverage(expected, observations),
        "latency_p50_ms": percentile([item.latency_ms for item in values], 50),
        "latency_p95_ms": percentile([item.latency_ms for item in values], 95),
    }


def _assert_baseline(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    for mode in set(report["retrieval"]).intersection(baseline["retrieval"]):
        current = report["retrieval"][mode]
        expected = baseline["retrieval"][mode]
        if current["recall_at_5"] < expected["recall_at_5"] - 0.05:
            raise SystemExit(f"{mode} Recall@5 regressed by more than 5 percentage points")
        if current["mrr"] < expected["mrr"] - 0.05:
            raise SystemExit(f"{mode} MRR regressed by more than 0.05")


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        f"- Documents: {report['dataset']['documents']}",
        f"- Cases: {report['dataset']['cases']}",
        "",
        "## Retrieval",
        "",
        "| Mode | Recall@5 | MRR |",
        "| --- | ---: | ---: |",
    ]
    for mode, metrics in report["retrieval"].items():
        lines.append(f"| {mode} | {metrics['recall_at_5']:.3f} | {metrics['mrr']:.3f} |")
    if "generation" in report:
        lines.extend(["", "## Generation", ""])
        lines.extend(f"- {name}: {value:.3f}" for name, value in report["generation"].items())
    lines.extend(
        [
            "",
            "The live modes are non-deterministic. Compare changes against the",
            "versioned baseline and inspect case-level failures before accepting them.",
            "",
        ]
    )
    return "\n".join(lines)


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/dataset.json"))
    parser.add_argument("--baseline", type=Path, default=Path("evals/baseline.json"))
    parser.add_argument("--ollama-url")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--api-url")
    parser.add_argument("--output", type=Path, default=Path("evals/report.json"))
    args = parser.parse_args()

    documents, cases = _load_dataset(args.dataset)
    retrieval = await _evaluate_retrieval(documents, cases, args.ollama_url, args.embedding_model)
    report: dict[str, Any] = {
        "dataset": {
            "documents": len(documents),
            "cases": len(cases),
        },
        "retrieval": retrieval,
    }
    if args.api_url:
        report["generation"] = await _evaluate_api(args.api_url, documents, cases)
    if args.baseline.exists():
        _assert_baseline(report, json.loads(args.baseline.read_text()))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(_markdown_report(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
