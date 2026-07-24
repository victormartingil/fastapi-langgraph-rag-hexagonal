"""Composition root: the ONE place where concrete adapters are chosen.

The whole application is wired together here:

- `build_container` creates the long-lived objects (engine, HTTP clients,
  provider adapters) from `Settings`.
- `provide_*` functions are FastAPI dependencies: they assemble per-request
  use cases from a fresh database session + the long-lived adapters.

Everywhere else, code depends on Protocols. If you want to know "which
implementation is actually used?", this file is the entire answer.
(Spring developers: this module plays the role of `@Configuration`.)
"""

import secrets
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated

import httpx
import structlog
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

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
from knowledge_assistant.assistant.application.ports import AnswerGenerator
from knowledge_assistant.config import Settings
from knowledge_assistant.knowledge_base.adapters.outbound.embeddings.ollama import (
    OllamaEmbeddingProvider,
)
from knowledge_assistant.knowledge_base.adapters.outbound.embeddings.openai import (
    OpenAiEmbeddingProvider,
)
from knowledge_assistant.knowledge_base.adapters.outbound.extraction.pdf import PdfTextExtractor
from knowledge_assistant.knowledge_base.adapters.outbound.extraction.plain_text import (
    PlainTextExtractor,
)
from knowledge_assistant.knowledge_base.adapters.outbound.persistence.repository import (
    SqlAlchemyDocumentRepository,
)
from knowledge_assistant.knowledge_base.adapters.outbound.retrieval.pgvector import (
    PgVectorRetriever,
)
from knowledge_assistant.knowledge_base.application.exceptions import (
    KnowledgeBaseUnavailableError,
)
from knowledge_assistant.knowledge_base.application.ingest import IngestDocument
from knowledge_assistant.knowledge_base.application.ports import (
    DocumentRepository,
    EmbeddingProvider,
    KnowledgeRetriever,
    OpenKnowledgeRetriever,
    OpenRepository,
    TextExtractor,
)
from knowledge_assistant.knowledge_base.application.queries import (
    GetDocument,
    ListDocuments,
    SearchKnowledge,
)
from knowledge_assistant.platform.database.session import (
    create_engine,
    create_session_factory,
    is_db_outage_error,
    session_scope,
)

logger = structlog.get_logger()


@dataclass
class Container:
    """Long-lived dependencies, created once per process at startup."""

    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    embedding_provider: EmbeddingProvider
    answer_generator: AnswerGenerator
    text_extractors: Sequence[TextExtractor]
    embedding_config: "ResolvedEmbeddingConfig"
    _embedding_http_client: httpx.AsyncClient
    _llm_http_client: httpx.AsyncClient

    async def aclose(self) -> None:
        """Release resources on shutdown (called from the lifespan): both
        long-lived HTTP clients and the engine pool.

        Each resource is closed independently: one failing close must not
        skip the rest. Failures are logged loudly and re-raised as a group,
        so a broken shutdown is visible rather than silent.
        """
        failures: list[Exception] = []
        for name, close in (
            ("embedding_http_client", self._embedding_http_client.aclose),
            ("llm_http_client", self._llm_http_client.aclose),
            ("engine", self.engine.dispose),
        ):
            try:
                await close()
            except Exception as exc:
                logger.exception("container_close_failed", resource=name)
                failures.append(exc)
        if failures:
            raise ExceptionGroup("container shutdown had failures", failures)


# ---------------------------------------------------------------------------
# Provider-driven defaults. A bare `KA_EMBEDDING_PROVIDER=openai` /
# `KA_LLM_PROVIDER=openai` is enough: these defaults fill in whatever the
# environment did not override explicitly. Every value can still be
# overridden with its KA_* variable.
# ---------------------------------------------------------------------------
_EMBEDDING_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",  # native Ollama API
        "dimension": "768",
    },
    "openai": {
        "model": "text-embedding-3-small",
        "base_url": "https://api.openai.com",
        "dimension": "1536",
    },
}

_LLM_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "ollama": {
        "model": "qwen3.5:9b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",  # self-hosted Ollama ignores it
    },
    "openai": {
        "model": "gpt-4o-mini",
        "base_url": "https://api.openai.com/v1",
        "api_key": "",  # MUST be provided via KA_LLM_API_KEY
    },
}

# The dimension of the pgvector column created by the SHIPPED migrations
# (vector(768), nomic-embed-text). Switching to a model with a different
# dimension — e.g. OpenAI's text-embedding-3-small at 1536 — means
# regenerating the Alembic migration AND updating this constant, deliberately
# manual: the schema is the source of truth (ADR-0001).
SCHEMA_EMBEDDING_DIMENSION = 768


@dataclass(frozen=True, slots=True)
class ResolvedEmbeddingConfig:
    """The effective embedding configuration after provider defaults apply."""

    model: str
    base_url: str
    api_key: str
    dimension: int


def _resolve_embedding_config(settings: Settings) -> ResolvedEmbeddingConfig:
    defaults = _EMBEDDING_PROVIDER_DEFAULTS[settings.embedding_provider]
    if settings.embedding_provider == "openai" and not settings.embedding_api_key.strip():
        msg = "KA_EMBEDDING_API_KEY is required when KA_EMBEDDING_PROVIDER=openai"
        raise ValueError(msg)
    return ResolvedEmbeddingConfig(
        model=settings.embedding_model or defaults["model"],
        base_url=settings.embedding_base_url or defaults["base_url"],
        api_key=settings.embedding_api_key,
        dimension=settings.embedding_dimension or int(defaults["dimension"]),
    )


def _build_embedding_provider(
    provider_name: str, resolved: ResolvedEmbeddingConfig, timeout_seconds: float
) -> tuple[EmbeddingProvider, httpx.AsyncClient]:
    """Strategy selection: the SAME port, two interchangeable adapters,
    chosen by configuration — no if/else anywhere else in the codebase."""
    client = httpx.AsyncClient(
        base_url=resolved.base_url,
        timeout=timeout_seconds,
    )
    provider: EmbeddingProvider
    if provider_name == "openai":
        provider = OpenAiEmbeddingProvider(
            client,
            model=resolved.model,
            api_key=resolved.api_key,
        )
    else:
        provider = OllamaEmbeddingProvider(client, model=resolved.model)
    return provider, client


def _build_answer_generator(settings: Settings) -> tuple[AnswerGenerator, httpx.AsyncClient]:
    """Strategy selection, symmetric with `_build_embedding_provider`: the
    provider flag drives the adapter's configuration. A single adapter class
    is the honest design here — both providers speak the OpenAI-compatible
    chat API; what differs is where it points and which model it calls.

    Like the embedding client, the LLM client is created HERE so its lifetime
    is owned by the Container (closed in `aclose`).
    """
    defaults = _LLM_PROVIDER_DEFAULTS[settings.llm_provider]
    api_key = settings.llm_api_key or defaults["api_key"]
    if settings.llm_provider == "openai" and not api_key.strip():
        msg = "KA_LLM_API_KEY is required when KA_LLM_PROVIDER=openai"
        raise ValueError(msg)
    # KA_LLM_TIMEOUT_SECONDS becomes the client timeout: an unbounded LLM
    # call would hold the request (and its database session) open forever.
    client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)
    generator = PydanticAiAnswerGenerator(
        provider=settings.llm_provider,
        model_name=settings.llm_model or defaults["model"],
        base_url=settings.llm_base_url or defaults["base_url"],
        api_key=api_key,
        http_client=client,
        max_retries=settings.llm_max_retries,
        output_retries=settings.llm_output_retries,
    )
    return generator, client


def build_container(settings: Settings) -> Container:
    """Create all long-lived dependencies from configuration."""
    embedding_config = _resolve_embedding_config(settings)
    if embedding_config.dimension != SCHEMA_EMBEDDING_DIMENSION:
        # The shipped schema is vector(768) (nomic-embed-text). Persisting a
        # vector of any other size would fail — or worse, corrupt retrieval —
        # at query time, so we refuse to start instead (ADR-0001). The fix is
        # a new migration for the new dimension + bumping
        # SCHEMA_EMBEDDING_DIMENSION with it, NOT widening this check.
        msg = (
            f"Embedding dimension {embedding_config.dimension} (provider "
            f"{settings.embedding_provider!r}, model {embedding_config.model!r}) does not "
            f"match the migrated schema (vector({SCHEMA_EMBEDDING_DIMENSION})). Regenerate "
            "the Alembic migration for the new dimension, update "
            "SCHEMA_EMBEDDING_DIMENSION in bootstrap.py, and re-embed the corpus — "
            "see docs/adr/0001."
        )
        raise ValueError(msg)

    # Validation before resource creation: a missing LLM key (raised inside
    # _build_answer_generator) must not leave an engine pool behind.
    answer_generator, llm_client = _build_answer_generator(settings)

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    embedding_provider, embedding_client = _build_embedding_provider(
        settings.embedding_provider, embedding_config, settings.embedding_timeout_seconds
    )

    return Container(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        embedding_provider=embedding_provider,
        answer_generator=answer_generator,
        text_extractors=(PlainTextExtractor(), PdfTextExtractor()),
        embedding_config=embedding_config,
        _embedding_http_client=embedding_client,
        _llm_http_client=llm_client,
    )


# ---------------------------------------------------------------------------
# FastAPI dependency providers (the "Depends" half of the DI story)
# ---------------------------------------------------------------------------


def get_container(request: Request) -> Container:
    # `app.state` is untyped by design (Starlette); the lifespan is what
    # guarantees this attribute exists and is a Container.
    container: Container = request.app.state.container
    return container


async def require_api_key(
    request: Request,
    container: Annotated[Container, Depends(get_container)],
) -> None:
    """Optional API-key guard for /api/v1/*.

    Off by default: when `KA_API_KEY` is unset this is a no-op and the
    zero-friction quick start keeps working. When set, every API request must
    carry a matching `X-API-Key` header. Deliberately simple — rate limiting
    and JWT/OIDC are roadmap items, not stealth complexity.
    """
    expected = container.settings.api_key
    if expected is None:
        return
    provided = request.headers.get("X-API-Key")
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key (send it in the X-API-Key header)",
            # RFC 9110 §11.6.1: a 401 MUST carry a WWW-Authenticate
            # challenge — `ApiKey` is our custom scheme name, realm scopes
            # it. Without it, conforming clients cannot recover.
            headers={"WWW-Authenticate": 'ApiKey realm="api"'},
        )


async def provide_session(
    container: Annotated[Container, Depends(get_container)],
) -> AsyncIterator[AsyncSession]:
    """One session (and transaction) per request; commit on success."""
    async with session_scope(container.session_factory) as session:
        yield session


def repository_scope_factory(
    session_factory: async_sessionmaker[AsyncSession],
) -> OpenRepository:
    """The `OpenRepository` port, wired to the pool: each call opens a SHORT
    unit of work (session + transaction via `session_scope`) around a real
    SqlAlchemyDocumentRepository.

    `IngestDocument` opens several of these per execution (dedup check /
    save / race recovery) instead of holding one session across the slow
    extraction+embedding steps — a pooled connection is pinned for
    milliseconds, not seconds (ADR-0005).

    A connection-level failure at COMMIT time escapes the repository's own
    translation (the commit lives in `session_scope`, after the repository
    method returned), so the scope re-applies the same outage doctrine
    (is_db_outage_error -> KnowledgeBaseUnavailableError, 503).
    DuplicateDocumentError — the race signal the use case recovers from —
    is a DomainError and passes through untouched.
    """

    @asynccontextmanager
    async def _scope() -> AsyncIterator[DocumentRepository]:
        try:
            async with session_scope(session_factory) as session:
                yield SqlAlchemyDocumentRepository(session)
        except Exception as exc:
            if is_db_outage_error(exc):
                raise KnowledgeBaseUnavailableError() from exc
            raise

    return _scope


def retriever_scope_factory(
    session_factory: async_sessionmaker[AsyncSession],
    embedding_provider: EmbeddingProvider,
    *,
    fetch_limit: int,
    rrf_k: int,
    tsconfig: str,
) -> OpenKnowledgeRetriever:
    """Open a retrieval adapter around one short read-side unit of work.

    Chat may spend most of its time in grading/generation. This factory keeps
    PostgreSQL scoped to the actual retrieval query, then releases the session
    before the assistant workflow continues to the LLM (ADR-0005).
    """

    @asynccontextmanager
    async def _scope() -> AsyncIterator[KnowledgeRetriever]:
        try:
            async with session_scope(session_factory) as session:
                yield PgVectorRetriever(
                    session,
                    embedding_provider,
                    fetch_limit=fetch_limit,
                    rrf_k=rrf_k,
                    tsconfig=tsconfig,
                )
        except Exception as exc:
            if is_db_outage_error(exc):
                raise KnowledgeBaseUnavailableError() from exc
            raise

    return _scope


def provide_ingest_document(
    container: Annotated[Container, Depends(get_container)],
) -> IngestDocument:
    """Assemble the ingest use case around the repository-SCOPE factory.

    Note what is NOT here: `Depends(provide_session)`. Ingest manages its
    own short transactions (ADR-0005), so it does not consume the
    request-scoped session at all.
    """
    return IngestDocument(
        open_repository=repository_scope_factory(container.session_factory),
        embedding_provider=container.embedding_provider,
        text_extractors=container.text_extractors,
        chunk_max_chars=container.settings.chunk_max_chars,
        chunk_overlap_chars=container.settings.chunk_overlap_chars,
        embedding_batch_size=container.settings.embedding_batch_size,
        expected_embedding_dimension=container.embedding_config.dimension,
        embedding_model_name=container.embedding_config.model,
    )


def provide_get_document(
    session: Annotated[AsyncSession, Depends(provide_session)],
) -> GetDocument:
    return GetDocument(repository=SqlAlchemyDocumentRepository(session))


def provide_list_documents(
    session: Annotated[AsyncSession, Depends(provide_session)],
) -> ListDocuments:
    return ListDocuments(repository=SqlAlchemyDocumentRepository(session))


def provide_ask_question(
    container: Annotated[Container, Depends(get_container)],
) -> AskQuestion:
    """Assemble the RAG graph for this request.

    The graph is compiled per request — cheap (no I/O), and it keeps the
    wiring honest. Retrieval gets a short database scope of its own, so the
    SQL session is closed before grading/generation can wait on the LLM.
    """
    settings = container.settings
    open_retriever = retriever_scope_factory(
        container.session_factory,
        container.embedding_provider,
        fetch_limit=settings.retrieval_fetch_limit,
        rrf_k=settings.rrf_k,
        tsconfig=settings.fts_language,
    )
    knowledge_search = InProcessKnowledgeSearchAdapter(SearchKnowledge(open_retriever))
    workflow = LangGraphRagWorkflow(
        knowledge_search,
        container.answer_generator,
        min_relevance_score=settings.min_relevance_score,
    )
    return AskQuestion(workflow, default_top_k=settings.retrieval_top_k)
