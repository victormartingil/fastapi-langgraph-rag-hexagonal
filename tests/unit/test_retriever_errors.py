"""Unit tests for the retriever's outage translation (adapter -> domain signal).

PgVectorHybridRetriever sits at the port boundary: like the repository
translating IntegrityError into DuplicateDocumentError, it translates
TRANSIENT infrastructure failures (embedding provider down, database
unreachable) into KnowledgeBaseUnavailableError, which the HTTP layer maps to
503. Permanent failures — a 401 from the provider, a SQL programming bug —
must NOT be translated: they are server errors and stay 500-visible.
"""

from typing import cast

import httpx
import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.knowledge_base.domain.exceptions import KnowledgeBaseUnavailableError
from knowledge_assistant.knowledge_base.infrastructure.retrieval.pgvector_hybrid import (
    PgVectorHybridRetriever,
)
from knowledge_assistant.shared.domain.exceptions import (
    EmbeddingProviderUnavailableError,
)
from knowledge_assistant.shared.domain.value_objects import EmbeddingVector
from tests.unit.fakes import FakeEmbeddingProvider

DIM = 768


class DownEmbeddingProvider:
    """Implements EmbeddingProvider by raising — a provider outage stub."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        raise self.error


class DownSession:
    """Stands in for AsyncSession, failing at execute() — a DB outage stub."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def execute(self, *args: object, **kwargs: object) -> object:
        raise self.error


def make_retriever(
    *,
    provider_error: Exception | None = None,
    session_error: Exception | None = None,
) -> PgVectorHybridRetriever:
    provider = (
        DownEmbeddingProvider(provider_error)
        if provider_error is not None
        else FakeEmbeddingProvider(dimension=DIM)
    )
    session = (
        DownSession(session_error)
        if session_error is not None
        else DownSession(AssertionError("the SQL leg should not be reached"))
    )
    # The stub quacks like the one method the retriever calls; the cast keeps
    # mypy honest without dragging a real engine into the unit tier.
    return PgVectorHybridRetriever(cast("AsyncSession", session), provider)


class TestProviderOutageTranslation:
    async def test_transient_provider_failure_becomes_retrieval_unavailable(self) -> None:
        retriever = make_retriever(provider_error=httpx.ConnectError("connection refused"))

        with pytest.raises(KnowledgeBaseUnavailableError, match="temporarily unavailable"):
            await retriever.retrieve("q?", limit=5)

    async def test_port_contract_error_is_rewrapped_as_the_chat_signal(self) -> None:
        """Real adapters honor the port contract (EmbeddingProviderUnavailableError);
        the assistant context re-wraps it as its OWN domain signal so contexts
        stay decoupled and the HTTP layer sees KnowledgeBaseUnavailableError."""
        retriever = make_retriever(
            provider_error=EmbeddingProviderUnavailableError("provider down")
        )

        with pytest.raises(KnowledgeBaseUnavailableError, match="temporarily unavailable"):
            await retriever.retrieve("q?", limit=5)

    async def test_permanent_provider_failure_is_not_translated(self) -> None:
        # A 401 is a misconfiguration, not an outage: translating it to 503
        # would tell the client "retry" when retrying can never help.
        request = httpx.Request("POST", "http://ollama.test/api/embed")
        response = httpx.Response(401, request=request)
        error = httpx.HTTPStatusError("unauthorized", request=request, response=response)
        retriever = make_retriever(provider_error=error)

        with pytest.raises(httpx.HTTPStatusError):
            await retriever.retrieve("q?", limit=5)


class TestDatabaseOutageTranslation:
    async def test_connection_level_failure_becomes_retrieval_unavailable(self) -> None:
        error = OperationalError("SELECT 1", {}, Exception("connection refused"))
        retriever = make_retriever(session_error=error)

        with pytest.raises(KnowledgeBaseUnavailableError, match="database unreachable"):
            await retriever.retrieve("q?", limit=5)

    async def test_raw_asyncpg_transport_failure_is_also_an_outage(self) -> None:
        """asyncpg's connect path raises a bare OSError (SQLAlchemy only
        wraps DBAPI errors) — classified as an outage just the same."""
        retriever = make_retriever(
            session_error=OSError("Multiple exceptions: [Errno 61] Connect call failed")
        )

        with pytest.raises(KnowledgeBaseUnavailableError, match="database unreachable"):
            await retriever.retrieve("q?", limit=5)

    async def test_sql_bugs_are_not_translated(self) -> None:
        # A programming error (bad SQL, missing column) is a 500-class bug;
        # reporting it as a transient 503 would hide it behind retries.
        retriever = make_retriever(session_error=AssertionError("bug"))

        with pytest.raises(AssertionError, match="bug"):
            await retriever.retrieve("q?", limit=5)
