"""RAG evaluation CLI, intentionally separate from deterministic tests.

Modes:

- ``deterministic``: offline lexical baseline for fast CI.
- ``live-retrieval``: real chunker, Ollama embeddings, PostgreSQL/pgvector,
  dense SQL, FTS SQL and hybrid RRF SQL.
- ``live-full``: live retrieval plus LangGraph grading and Ollama generation.

Reports store metrics and case ids only. Prompts, chunks and generated text
stay out of artifacts by default.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_assistant.assistant.adapters.outbound.knowledge.in_process import (
    InProcessKnowledgeSearchAdapter,
)
from knowledge_assistant.assistant.adapters.outbound.llm.pydantic_ai import (
    PydanticAiAnswerGenerator,
)
from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph.builder import (
    LangGraphRagWorkflow,
)
from knowledge_assistant.assistant.application.ask import AskQuestion
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
from knowledge_assistant.knowledge_base.adapters.outbound.embeddings.ollama import (
    OllamaEmbeddingProvider,
)
from knowledge_assistant.knowledge_base.adapters.outbound.persistence.repository import (
    SqlAlchemyDocumentRepository,
)
from knowledge_assistant.knowledge_base.adapters.outbound.retrieval.pgvector import (
    PgVectorRetriever,
)
from knowledge_assistant.knowledge_base.application.queries import SearchKnowledge
from knowledge_assistant.knowledge_base.application.retrieval import RetrievalStrategy
from knowledge_assistant.knowledge_base.domain.chunking import chunk_text
from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import ChunkId, ChunkText, DocumentId
from knowledge_assistant.platform.database.session import (
    create_engine,
    create_session_factory,
    session_scope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PGVECTOR_IMAGE = (
    "pgvector/pgvector:0.8.1-pg16@"
    "sha256:33198da2828a14c30348d2ccb4750833d5ed9a44c88d840a0e523d7417120337"
)
EVAL_NAMESPACE = uuid.UUID("12b02b69-69c4-5f1a-b560-5ad861c1d661")
TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
EvalMode = Literal["deterministic", "live-retrieval", "live-full"]


@dataclass(frozen=True, slots=True)
class EvaluationEnvironment:
    mode: EvalMode
    embedding_model: str
    llm_model: str | None
    ollama_url: str | None
    pgvector_image: str | None


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


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_PATTERN.findall(text)}


def _load_dataset(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = json.loads(path.read_text())
    return payload["documents"], payload["cases"]


def _expected(cases: Sequence[Mapping[str, Any]]) -> list[ExpectedCase]:
    return [
        ExpectedCase(
            case_id=str(case["id"]),
            answerable=bool(case["answerable"]),
            relevant_documents=frozenset(str(item) for item in case["relevant_documents"]),
            expected_facts=tuple(str(fact) for fact in case.get("expected_facts", ())),
        )
        for case in cases
    ]


def _retrieval_metrics(
    cases: Sequence[ExpectedCase],
    rankings: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    return {
        "recall_at_5": recall_at_k(cases, rankings, 5),
        "mrr": mean_reciprocal_rank(cases, rankings),
    }


def _case_retrieval_results(
    cases: Sequence[ExpectedCase],
    rankings: Mapping[str, Sequence[str]],
) -> list[dict[str, object]]:
    by_id = {case.case_id: case for case in cases}
    return [
        {
            "case_id": case_id,
            "answerable": by_id[case_id].answerable,
            "found_relevant": bool(by_id[case_id].relevant_documents.intersection(ranking[:5])),
            "ranked_documents": list(ranking[:5]),
        }
        for case_id, ranking in sorted(rankings.items())
    ]


def _evaluate_deterministic_retrieval(
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, object]:
    expected = _expected(cases)
    rankings = {str(case["id"]): lexical_rank(str(case["question"]), documents) for case in cases}
    return {
        "metrics": {"lexical": _retrieval_metrics(expected, rankings)},
        "cases": {"lexical": _case_retrieval_results(expected, rankings)},
    }


async def _evaluate_live_retrieval(
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    ollama_url: str,
    embedding_model: str,
    pgvector_image: str,
) -> dict[str, object]:
    async with (
        _live_database(pgvector_image) as session_factory,
        httpx.AsyncClient(base_url=ollama_url, timeout=180) as client,
    ):
        embedding_provider = OllamaEmbeddingProvider(client, embedding_model, max_retries=2)
        await _ingest_documents(
            session_factory,
            embedding_provider,
            documents,
        )
        return await _evaluate_pgvector_strategies(
            session_factory,
            embedding_provider,
            documents,
            cases,
        )


async def _evaluate_pgvector_strategies(
    session_factory: async_sessionmaker[AsyncSession],
    embedding_provider: OllamaEmbeddingProvider,
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, object]:
    expected = _expected(cases)
    title_to_dataset_id = {str(document["title"]): str(document["id"]) for document in documents}
    all_rankings: dict[str, dict[str, list[str]]] = {}
    hybrid_hits: dict[str, list[tuple[str, float]]] = {}
    for strategy in RetrievalStrategy:
        rankings: dict[str, list[str]] = {}
        async with session_scope(session_factory) as session:
            retriever = PgVectorRetriever(session, embedding_provider)
            for case in cases:
                hits = await retriever.retrieve(
                    str(case["question"]),
                    limit=5,
                    strategy=strategy,
                )
                case_id = str(case["id"])
                rankings[case_id] = _unique_document_ids(
                    title_to_dataset_id.get(hit.document_title, "<unknown>") for hit in hits
                )
                if strategy == RetrievalStrategy.HYBRID:
                    hybrid_hits[case_id] = [
                        (title_to_dataset_id.get(hit.document_title, "<unknown>"), hit.score)
                        for hit in hits
                    ]
        all_rankings[strategy.value] = rankings
    return {
        "metrics": {
            strategy: _retrieval_metrics(expected, rankings)
            for strategy, rankings in all_rankings.items()
        },
        "cases": {
            strategy: _case_retrieval_results(expected, rankings)
            for strategy, rankings in all_rankings.items()
        },
        "calibration": _calibrate_relevance_threshold(expected, hybrid_hits),
    }


async def _evaluate_live_full(
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    *,
    ollama_url: str,
    embedding_model: str,
    llm_model: str,
    pgvector_image: str,
) -> tuple[dict[str, object], dict[str, object]]:
    async with (
        _live_database(pgvector_image) as session_factory,
        httpx.AsyncClient(base_url=ollama_url, timeout=180) as embedding_client,
        httpx.AsyncClient(timeout=180) as llm_client,
    ):
        embedding_provider = OllamaEmbeddingProvider(
            embedding_client,
            embedding_model,
            max_retries=2,
        )
        await _ingest_documents(
            session_factory,
            embedding_provider,
            documents,
        )
        retrieval = await _evaluate_pgvector_strategies(
            session_factory,
            embedding_provider,
            documents,
            cases,
        )
        answer_generator = PydanticAiAnswerGenerator(
            provider="ollama",
            model_name=llm_model,
            base_url=f"{ollama_url.rstrip('/')}/v1",
            api_key="ollama",
            http_client=llm_client,
            max_retries=1,
            output_retries=1,
        )
        ask_question = _build_ask_question(
            session_factory,
            embedding_provider,
            answer_generator,
        )
        generation = await _evaluate_generation(ask_question, documents, cases)
    return retrieval, generation


def _build_ask_question(
    session_factory: async_sessionmaker[AsyncSession],
    embedding_provider: OllamaEmbeddingProvider,
    answer_generator: PydanticAiAnswerGenerator,
) -> AskQuestion:
    @asynccontextmanager
    async def open_retriever() -> AsyncIterator[PgVectorRetriever]:
        async with session_scope(session_factory) as session:
            yield PgVectorRetriever(session, embedding_provider)

    knowledge_search = InProcessKnowledgeSearchAdapter(SearchKnowledge(open_retriever))
    workflow = LangGraphRagWorkflow(knowledge_search, answer_generator, min_relevance_score=0.028)
    return AskQuestion(workflow, default_top_k=5)


async def _evaluate_generation(
    ask_question: AskQuestion,
    documents: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, object]:
    known_documents = frozenset(str(document["id"]) for document in documents)
    title_to_id = {str(document["title"]): str(document["id"]) for document in documents}
    observations: dict[str, AnswerObservation] = {}
    case_rows: list[dict[str, object]] = []
    for case in cases:
        started = time.perf_counter()
        answer = await ask_question.execute(str(case["question"]), top_k=5)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        cited = tuple(
            title_to_id.get(source.document_title, "<unknown>") for source in answer.sources
        )
        observations[str(case["id"])] = AnswerObservation(
            refused=not cited,
            cited_documents=cited,
            known_documents=known_documents,
            answer_text=answer.text,
            latency_ms=elapsed_ms,
        )
        expected_case = _expected((case,))[0]
        case_rows.append(
            {
                "case_id": str(case["id"]),
                "answerable": expected_case.answerable,
                "refused": not cited,
                "cited_documents": list(cited),
                "citation_valid": all(document_id in known_documents for document_id in cited),
                "fact_coverage": _single_case_fact_coverage(expected_case, answer.text),
                "latency_ms": elapsed_ms,
            }
        )
    expected = _expected(cases)
    values = list(observations.values())
    return {
        "metrics": {
            "abstention_accuracy": abstention_accuracy(expected, observations),
            "citation_validity": citation_validity(values),
            "fact_coverage": fact_coverage(expected, observations),
            "latency_p50_ms": percentile([item.latency_ms for item in values], 50),
            "latency_p95_ms": percentile([item.latency_ms for item in values], 95),
        },
        "cases": case_rows,
    }


def _single_case_fact_coverage(case: ExpectedCase, answer_text: str) -> float | None:
    if not case.answerable or not case.expected_facts:
        return None
    return sum(fact.casefold() in answer_text.casefold() for fact in case.expected_facts) / len(
        case.expected_facts
    )


async def _ingest_documents(
    session_factory: async_sessionmaker[AsyncSession],
    embedding_provider: OllamaEmbeddingProvider,
    documents: Sequence[Mapping[str, Any]],
) -> None:
    async with session_scope(session_factory) as session:
        repository = SqlAlchemyDocumentRepository(session)
        for item in documents:
            raw_text = str(item["text"])
            chunk_texts = chunk_text(raw_text)
            embeddings = await embedding_provider.embed([str(text) for text in chunk_texts])
            document_id = _stable_document_id(str(item["id"]))
            await repository.save(
                Document(
                    id=document_id,
                    title=str(item["title"]),
                    file_name=f"{item['id']}.txt",
                    raw_text=raw_text,
                    chunks=tuple(
                        Chunk(
                            id=_stable_chunk_id(str(item["id"]), str(position)),
                            text=ChunkText(str(text)),
                            position=position,
                            embedding=embeddings[position],
                        )
                        for position, text in enumerate(chunk_texts)
                    ),
                    content_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
                )
            )


def _stable_document_id(*parts: str) -> DocumentId:
    return DocumentId(
        uuid.uuid5(
            EVAL_NAMESPACE,
            ":".join(parts),
        )
    )


def _stable_chunk_id(*parts: str) -> ChunkId:
    return ChunkId(
        uuid.uuid5(
            EVAL_NAMESPACE,
            ":".join(parts),
        )
    )


@asynccontextmanager
async def _live_database(
    image: str,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(image) as postgres:
        host = postgres.get_container_host_ip()
        port = postgres.get_exposed_port(5432)
        database_url = (
            f"postgresql+asyncpg://{postgres.username}:{postgres.password}"
            f"@{host}:{port}/{postgres.dbname}"
        )
        _run_migrations(database_url)
        engine = create_engine(database_url)
        try:
            yield create_session_factory(engine)
        finally:
            await engine.dispose()


def _run_migrations(database_url: str) -> None:
    alembic = shutil.which("alembic") or str(Path(sys.executable).parent / "alembic")
    env = {**os.environ, "KA_DATABASE_URL": database_url}
    try:
        subprocess.run(
            [alembic, "upgrade", "head"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Alembic migrations failed against evaluation database:\n{exc.stdout}\n{exc.stderr}"
        ) from exc


def _unique_document_ids(document_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for document_id in document_ids:
        value = str(document_id)
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _calibrate_relevance_threshold(
    cases: Sequence[ExpectedCase],
    hybrid_hits: Mapping[str, Sequence[tuple[str, float]]],
) -> dict[str, object]:
    all_scores = {score for hits in hybrid_hits.values() for _, score in hits}
    unanswerable_scores = {
        score
        for case in cases
        if not case.answerable
        for _, score in hybrid_hits.get(case.case_id, ())
    }
    # Include the first threshold that is strictly above the strongest
    # unanswerable hit; otherwise an equality comparison can never produce
    # zero false positives when the best unanswerable score is itself a
    # candidate.
    if unanswerable_scores:
        all_scores.add(max(unanswerable_scores) + 1e-12)
    candidate_scores = sorted(all_scores, reverse=True)
    best: dict[str, object] | None = None
    for threshold in candidate_scores:
        false_positives = sum(
            (not case.answerable)
            and any(score >= threshold for _, score in hybrid_hits.get(case.case_id, ()))
            for case in cases
        )
        answerable = [case for case in cases if case.answerable]
        recall = (
            sum(
                any(
                    document_id in case.relevant_documents and score >= threshold
                    for document_id, score in hybrid_hits.get(case.case_id, ())
                )
                for case in answerable
            )
            / len(answerable)
            if answerable
            else 0.0
        )
        candidate: dict[str, object] = {
            "threshold": threshold,
            "answerable_recall": recall,
            "false_positives": false_positives,
            "meets_minimum_answerable_recall": recall >= 0.85,
        }
        if false_positives == 0 and (
            best is None
            or recall > cast(float, best["answerable_recall"])
            or (recall == best["answerable_recall"] and threshold > cast(float, best["threshold"]))
        ):
            best = candidate
    return best or {
        "threshold": None,
        "answerable_recall": 0.0,
        "false_positives": None,
        "meets_minimum_answerable_recall": False,
    }


def _retrieval_section(report: Mapping[str, Any]) -> Mapping[str, Any]:
    retrieval = cast(Mapping[str, Any], report["retrieval"])
    if "metrics" in retrieval:
        return cast(Mapping[str, Any], retrieval["metrics"])
    return retrieval


def _assert_baseline(report: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
    current_retrieval = _retrieval_section(report)
    expected_retrieval = _retrieval_section(baseline)
    for mode in set(current_retrieval).intersection(expected_retrieval):
        current = current_retrieval[mode]
        expected = expected_retrieval[mode]
        if current["recall_at_5"] < expected["recall_at_5"] - 0.05:
            raise SystemExit(f"{mode} Recall@5 regressed by more than 5 percentage points")
        if current["mrr"] < expected["mrr"] - 0.05:
            raise SystemExit(f"{mode} MRR regressed by more than 0.05")


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# RAG evaluation report",
        "",
        f"- Mode: {report['environment']['mode']}",
        f"- Documents: {report['dataset']['documents']}",
        f"- Cases: {report['dataset']['cases']}",
        f"- Embedding model: {report['environment']['embedding_model']}",
        f"- LLM model: {report['environment'].get('llm_model') or 'not used'}",
        "",
        "## Retrieval",
        "",
        "| Mode | Recall@5 | MRR |",
        "| --- | ---: | ---: |",
    ]
    for mode, metrics in _retrieval_section(report).items():
        lines.append(f"| {mode} | {metrics['recall_at_5']:.3f} | {metrics['mrr']:.3f} |")
    if "generation" in report:
        generation_metrics = cast(Mapping[str, float], report["generation"]["metrics"])
        lines.extend(["", "## Generation", ""])
        lines.extend(f"- {name}: {value:.3f}" for name, value in generation_metrics.items())
    calibration = report["retrieval"].get("calibration")
    if calibration:
        lines.extend(
            [
                "",
                "## Retrieval threshold calibration",
                "",
                f"- threshold: {calibration['threshold']}",
                f"- answerable recall: {calibration['answerable_recall']:.3f}",
                f"- false positives: {calibration['false_positives']}",
            ]
        )
    lines.extend(
        [
            "",
            "Live modes are infrastructure- and model-dependent. Treat regressions",
            "as review gates, then inspect case-level failures before accepting them.",
            "",
        ]
    )
    return "\n".join(lines)


async def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    documents, cases = _load_dataset(args.dataset)
    mode = cast(EvalMode, args.mode)
    environment = EvaluationEnvironment(
        mode=mode,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model if mode == "live-full" else None,
        ollama_url=args.ollama_url if mode != "deterministic" else None,
        pgvector_image=args.pgvector_image if mode != "deterministic" else None,
    )
    generation: dict[str, object] | None = None
    if mode == "deterministic":
        retrieval = _evaluate_deterministic_retrieval(documents, cases)
    elif mode == "live-retrieval":
        retrieval = await _evaluate_live_retrieval(
            documents,
            cases,
            ollama_url=args.ollama_url,
            embedding_model=args.embedding_model,
            pgvector_image=args.pgvector_image,
        )
    else:
        retrieval, generation = await _evaluate_live_full(
            documents,
            cases,
            ollama_url=args.ollama_url,
            embedding_model=args.embedding_model,
            llm_model=args.llm_model,
            pgvector_image=args.pgvector_image,
        )
    report: dict[str, Any] = {
        "dataset": {
            "documents": len(documents),
            "cases": len(cases),
            "version": json.loads(args.dataset.read_text()).get("version"),
        },
        "environment": {
            "mode": environment.mode,
            "embedding_model": environment.embedding_model,
            "llm_model": environment.llm_model,
            "ollama_url": environment.ollama_url,
            "pgvector_image": environment.pgvector_image,
        },
        "retrieval": retrieval,
    }
    if generation is not None:
        report["generation"] = generation
    return report


async def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("deterministic", "live-retrieval", "live-full"),
        default="deterministic",
    )
    parser.add_argument("--dataset", type=Path, default=Path("evals/dataset.json"))
    parser.add_argument(
        "--baseline",
        type=Path,
        help=(
            "Baseline JSON to compare against. Defaults to evals/baseline.json "
            "only in deterministic mode; live baselines must be passed explicitly."
        ),
    )
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--llm-model", default="qwen3.5:9b")
    parser.add_argument("--pgvector-image", default=DEFAULT_PGVECTOR_IMAGE)
    parser.add_argument("--output", type=Path, default=Path("evals/report.json"))
    args = parser.parse_args()

    report = await _build_report(args)
    baseline = args.baseline or (
        Path("evals/baseline.json") if args.mode == "deterministic" else None
    )
    if baseline is not None and baseline.exists():
        _assert_baseline(report, json.loads(baseline.read_text()))
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.output.with_suffix(".md").write_text(_markdown_report(report))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
