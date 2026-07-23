"""Unit tests for the repository's database-outage translation.

Symmetric with the retriever's translation on the read path
(test_retriever_errors.py): a DEAD database (SQLAlchemy's
OperationalError/InterfaceError — connection refused, dropped, partitioned)
becomes the knowledge-base context's own 503-class domain signal,
KnowledgeBaseUnavailableError. SQL BUGS (ProgrammingError) are NOT
translated: they are 500-class defects and must stay loud.
"""

from typing import cast

import pytest
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_assistant.knowledge_base.adapters.outbound.persistence.repository import (
    SqlAlchemyDocumentRepository,
)
from knowledge_assistant.knowledge_base.application.exceptions import (
    KnowledgeBaseUnavailableError,
)
from knowledge_assistant.knowledge_base.domain.models import Document
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId

CONNECTION_REFUSED = OperationalError("SELECT 1", {}, Exception("connection refused"))
CONNECTION_DROPPED = InterfaceError("SELECT 1", {}, Exception("connection dropped"))
# asyncpg's connect path does NOT raise DBAPI errors: "connection refused"
# escapes SQLAlchemy's translation as a bare OSError (verified live against
# a dead port). It must be classified as an outage just the same.
RAW_TRANSPORT_FAILURE = OSError("Multiple exceptions: [Errno 61] Connect call failed")
SQL_BUG = ProgrammingError("SELECT nope", {}, Exception('column "nope" does not exist'))


class DownSession:
    """Stands in for AsyncSession, failing at flush/execute — a DB outage
    (or bug) stub. Only the methods the repository actually calls exist."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def add(self, instance: object) -> None:
        pass  # unit-of-work bookkeeping; the failure happens at flush

    async def flush(self) -> None:
        raise self.error

    async def rollback(self) -> None:
        pass

    async def execute(self, *args: object, **kwargs: object) -> object:
        raise self.error


def make_repository(error: Exception) -> SqlAlchemyDocumentRepository:
    # The stub quacks like the methods the repository calls; the cast keeps
    # mypy honest without dragging a real engine into the unit tier.
    return SqlAlchemyDocumentRepository(cast("AsyncSession", DownSession(error)))


def _document() -> Document:
    return Document(
        id=DocumentId(),
        title="Policy",
        file_name="policy.md",
        raw_text="Some text.",
        chunks=(),
        content_hash="deadbeef" * 8,
    )


OUTAGE_ERRORS = [CONNECTION_REFUSED, CONNECTION_DROPPED, RAW_TRANSPORT_FAILURE]


class TestDatabaseOutageTranslation:
    @pytest.mark.parametrize("error", OUTAGE_ERRORS)
    async def test_connection_level_failure_on_write_becomes_kb_unavailable(
        self, error: Exception
    ) -> None:
        repository = make_repository(error)

        with pytest.raises(KnowledgeBaseUnavailableError, match="temporarily unavailable"):
            await repository.save(_document())

    @pytest.mark.parametrize("error", OUTAGE_ERRORS)
    async def test_connection_level_failure_on_read_becomes_kb_unavailable(
        self, error: Exception
    ) -> None:
        repository = make_repository(error)

        with pytest.raises(KnowledgeBaseUnavailableError, match="temporarily unavailable"):
            await repository.get_summary_by_id(DocumentId())

    async def test_sql_bugs_are_not_translated(self) -> None:
        # A programming error (bad SQL, missing column) is a 500-class bug;
        # reporting it as a transient 503 would hide it behind retries.
        repository = make_repository(SQL_BUG)

        with pytest.raises(ProgrammingError):
            await repository.get_summary_by_id(DocumentId())
