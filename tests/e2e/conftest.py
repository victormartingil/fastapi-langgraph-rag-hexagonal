"""E2E fixtures: the full HTTP stack against a real database, with faked AI.

What is REAL here: FastAPI routing, middleware, error handlers, the Alembic-
built PostgreSQL schema, the repository, the hybrid SQL, the LangGraph graph.

What is FAKED (via FastAPI dependency overrides — the same seam used in
production to swap vendors): the embedding provider and the answer generator.
That keeps e2e deterministic and offline while still exercising the entire
request pipeline.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant import container as container_module
from knowledge_assistant.chat.application.graph.builder import build_rag_graph
from knowledge_assistant.chat.application.service import AskQuestion
from knowledge_assistant.chat.domain.exceptions import GenerationUnavailableError
from knowledge_assistant.chat.domain.models import Answer, RetrievedChunk, Source
from knowledge_assistant.chat.infrastructure.retrieval.pgvector_hybrid import (
    PgVectorHybridRetriever,
)
from knowledge_assistant.config import Settings
from knowledge_assistant.container import Container, build_container
from knowledge_assistant.documents.application.services import IngestDocument
from knowledge_assistant.documents.infrastructure.extraction.pdf import PdfTextExtractor
from knowledge_assistant.documents.infrastructure.extraction.plain_text import (
    PlainTextExtractor,
)
from knowledge_assistant.main import create_app
from knowledge_assistant.shared.domain.exceptions import (
    EmbeddingProviderUnavailableError,
)
from knowledge_assistant.shared.domain.value_objects import EmbeddingVector
from tests.unit.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

EMBEDDING_DIM = 768  # the migrated schema is vector(768)
SAMPLE_DOC = Path(__file__).resolve().parents[2] / "samples" / "return-policy.md"


class ScriptedAnswerGenerator:
    """Answers by echoing the chunks it was given — deterministic, offline,
    and still proving that retrieved evidence flows end-to-end into sources."""

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        return Answer(
            text=f"Grounded answer to: {question}",
            sources=tuple(
                Source(
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    chunk_id=chunk.chunk_id,
                    excerpt=chunk.content[:300],
                    score=chunk.score,
                )
                for chunk in chunks
            ),
        )


class UnreachableEmbeddingProvider:
    """Every call fails the way the REAL adapters fail after exhausted
    retries: with the port's contract error (EmbeddingProviderUnavailableError).

    Used with the real wiring (no dependency overrides) so both outage paths
    are exercised end to end: ingest -> 503 directly, chat -> re-wrapped as
    RetrievalUnavailableError -> 503.
    """

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        raise EmbeddingProviderUnavailableError(
            "The embedding service is temporarily unavailable "
            "(provider 'ollama' unreachable after retries). Please try again shortly."
        )


class UnavailableAnswerGenerator:
    """Every call fails the way the REAL adapter fails after exhausted
    transient retries: with the port's contract error (GenerationUnavailableError).
    """

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        raise GenerationUnavailableError(
            "The answer-generation service is temporarily unavailable "
            "(LLM unreachable after retries). Please try again shortly."
        )


@asynccontextmanager
async def _build_client(
    settings: Settings,
    *,
    swap_ai_ports_only: bool = False,
    provider_down: bool = False,
    llm_down: bool = False,
    db_down: bool = False,
    broken_pipeline: bool = False,
) -> AsyncIterator[AsyncClient]:
    """Assemble the full app around `settings` and yield a test HTTP client.

    All fixtures below share this body; they differ only in the Settings
    overrides — the same seam an operator uses via environment variables.

    `swap_ai_ports_only` chooses HOW the AI is faked:
    - False (default): FastAPI dependency overrides replace the whole
      per-request assembly (`provide_ingest_document` / `provide_ask_question`)
      with hand-built use cases around fakes.
    - True: the REAL container providers run (so settings wiring — chunking,
      batch size, fetch_limit, rrf_k, min_relevance_score, default top_k —
      is exercised end to end) and only the two AI ports are swapped on the
      container itself: fake embeddings in, scripted answers in.
    `provider_down` (implies the real wiring) installs an embedding provider
    that always fails, simulating an outage at query time.
    `db_down` points the whole stack at a dead database port (the engine
    connects lazily, so boot succeeds and every query fails at call time).
    `broken_pipeline` replaces a dependency with one that raises a plain
    RuntimeError — the unhandled-exception path (500 envelope)."""
    app = create_app(settings)

    # ASGITransport does not run the lifespan, so we build the container
    # manually — exactly what the lifespan does in production.
    container: Container = build_container(settings)
    app.state.container = container

    if not db_down:
        # The DATABASE is shared with the integration suite (one testcontainer
        # per pytest session): start from clean tables so tests never depend on
        # execution order. The container itself is rebuilt per fixture.
        # (Skipped for db_down: the whole point is that the DB refuses
        # connections, so a cleanup DELETE could never succeed.)
        async with container.session_factory() as session:
            await session.execute(text("DELETE FROM chunks"))
            await session.execute(text("DELETE FROM documents"))
            await session.commit()

    if swap_ai_ports_only or provider_down or llm_down or db_down:
        # Mutate the composition root's long-lived adapters; every per-request
        # provider below then assembles the REAL wiring around the fakes.
        container.embedding_provider = (
            UnreachableEmbeddingProvider()
            if provider_down
            else FakeEmbeddingProvider(dimension=EMBEDDING_DIM)
        )
        container.answer_generator = (
            UnavailableAnswerGenerator() if llm_down else ScriptedAnswerGenerator()
        )
    else:
        _install_dependency_overrides(app)

    if broken_pipeline:

        def broken_provide_ask_question() -> AskQuestion:
            raise RuntimeError("boom: an unhandled, undomained failure")

        app.dependency_overrides[container_module.provide_ask_question] = (
            broken_provide_ask_question
        )

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as http_client:
        yield http_client

    await container.aclose()


def _install_dependency_overrides(app: FastAPI) -> None:
    """Replace the AI boundary at the FastAPI seam: per-request providers are
    swapped for versions built around a fake embedder + scripted generator.
    The retriever stays REAL (real SQL, real pgvector)."""
    fake_embeddings = FakeEmbeddingProvider(dimension=EMBEDDING_DIM)

    def override_provide_ingest_document(
        container: Annotated[Container, Depends(container_module.get_container)],
    ) -> IngestDocument:
        # Same shape as the production provider (ADR-0005): the use case
        # gets a repository-SCOPE factory, not a request-scoped repository.
        return IngestDocument(
            open_repository=container_module.repository_scope_factory(container.session_factory),
            embedding_provider=fake_embeddings,
            text_extractors=(PlainTextExtractor(), PdfTextExtractor()),
        )

    def override_provide_ask_question(
        session: Annotated[AsyncSession, Depends(container_module.provide_session)],
    ) -> AskQuestion:
        retriever = PgVectorHybridRetriever(session, fake_embeddings)
        graph = build_rag_graph(
            retriever, ScriptedAnswerGenerator(), min_relevance_score=0.028
        ).compile()
        return AskQuestion(graph)

    app.dependency_overrides[container_module.provide_ingest_document] = (
        override_provide_ingest_document
    )
    app.dependency_overrides[container_module.provide_ask_question] = override_provide_ask_question


@pytest.fixture
async def client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """Default stack: no API key configured, generous upload limit."""
    async with _build_client(Settings(database_url=postgres_database_url)) as http_client:
        yield http_client


@pytest.fixture
async def limited_client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """Stack with a tiny upload limit (104 bytes) to exercise the 413 path."""
    settings = Settings(database_url=postgres_database_url, max_upload_size_mb=0.0001)
    async with _build_client(settings) as http_client:
        yield http_client


@pytest.fixture
async def authed_client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """Stack with API-key auth switched on (`X-API-Key: test-secret`)."""
    settings = Settings(database_url=postgres_database_url, api_key="test-secret")
    async with _build_client(settings) as http_client:
        yield http_client


@pytest.fixture
async def real_wiring_client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """The REAL per-request assembly (no dependency overrides), with only the
    two AI ports swapped on the container. `retrieval_top_k=1` makes the
    server-side default top_k observable: if the knob were not wired, the
    default would be 5 and the test could not tell."""
    settings = Settings(database_url=postgres_database_url, retrieval_top_k=1)
    async with _build_client(settings, swap_ai_ports_only=True) as http_client:
        yield http_client


@pytest.fixture
async def provider_down_client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """The REAL wiring with an embedding provider that is DOWN: both call
    sites must degrade honestly to 503 (ingest AND chat), never a bare 500."""
    settings = Settings(database_url=postgres_database_url)
    async with _build_client(settings, provider_down=True) as http_client:
        yield http_client


@pytest.fixture
async def llm_down_client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """The REAL wiring with an LLM that is DOWN at answer time: generation
    must degrade honestly to 503, never a degraded 200 fallback message."""
    settings = Settings(database_url=postgres_database_url)
    async with _build_client(settings, llm_down=True) as http_client:
        yield http_client


@pytest.fixture
async def db_down_client() -> AsyncIterator[AsyncClient]:
    """The REAL wiring with the DATABASE DOWN (dead port: connection refused).

    The engine connects lazily and the ASGI transport bypasses the lifespan,
    so the app boots fine — and every repository call then fails at query
    time. Every path must answer with the honest 503 outage envelope (or, for
    probes, the documented liveness/readiness behavior), never a bare 500."""
    settings = Settings(database_url="postgresql+asyncpg://postgres:postgres@localhost:1/knowledge")
    async with _build_client(settings, db_down=True) as http_client:
        yield http_client


@pytest.fixture
async def broken_client(postgres_database_url: str) -> AsyncIterator[AsyncClient]:
    """A pipeline stage that raises a plain RuntimeError — neither a domain
    error nor an HTTPException. The last line of defense must still answer
    with the unified 500 envelope and the correlation header."""
    settings = Settings(database_url=postgres_database_url)
    async with _build_client(settings, broken_pipeline=True) as http_client:
        yield http_client
