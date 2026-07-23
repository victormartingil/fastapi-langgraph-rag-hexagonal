"""SqlAlchemyDocumentRepository: the persistence adapter.

Implements the `DocumentRepository` Protocol — structurally, without
inheriting from it. All SQLAlchemy knowledge is quarantined in this file and
`models.py`; the application layer only sees domain objects and read models.

The session is injected per unit of work by the composition root;
transaction control (commit/rollback) lives in `session_scope`, not here.

Outage doctrine: a DEAD database (connection refused/dropped — SQLAlchemy's
OperationalError/InterfaceError) is translated into the domain signal
`KnowledgeBaseUnavailableError` (→ HTTP 503), symmetric with the retriever's
translation on the read path. SQL BUGS (ProgrammingError etc.) are NOT
translated: they are 500-class defects, not outages.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.selectable import Subquery

from knowledge_assistant.documents.application.read_models import DocumentSummary
from knowledge_assistant.documents.domain.exceptions import (
    DuplicateDocumentError,
    KnowledgeBaseUnavailableError,
)
from knowledge_assistant.documents.domain.models import Document
from knowledge_assistant.documents.domain.value_objects import DocumentId
from knowledge_assistant.documents.infrastructure.persistence.mappers import (
    document_to_domain,
    document_to_model,
    row_to_summary,
)
from knowledge_assistant.documents.infrastructure.persistence.models import (
    ChunkModel,
    DocumentModel,
)
from knowledge_assistant.shared.infrastructure.database import is_db_outage_error


@asynccontextmanager
async def _translate_db_outage() -> AsyncIterator[None]:
    """One translation for every public method: a connection-level failure
    becomes the domain's 503 signal; everything else (IntegrityError,
    ProgrammingError, bugs) escapes untouched."""
    try:
        yield
    except Exception as exc:
        if is_db_outage_error(exc):
            raise KnowledgeBaseUnavailableError() from exc
        raise


class SqlAlchemyDocumentRepository:
    """PostgreSQL-backed repository for the Document aggregate."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, document: Document) -> None:
        # `chunks` cascade from the aggregate root, so one add() is enough.
        async with _translate_db_outage():
            self._session.add(document_to_model(document))
            try:
                await self._session.flush()
            except IntegrityError as exc:
                # Two concurrent identical uploads can both pass the use case's
                # dedup check; the content_hash unique index is the last line of
                # defense (TOCTOU). Translate the vendor error into a domain
                # signal the use case can recover from — and roll back first,
                # because the failed flush has aborted the transaction.
                await self._session.rollback()
                if _is_content_hash_violation(exc):
                    raise DuplicateDocumentError(document.content_hash or "") from exc
                raise

    async def get_by_id(self, document_id: DocumentId) -> Document | None:
        async with _translate_db_outage():
            model = await self._session.get(DocumentModel, document_id.value)
        return document_to_domain(model) if model is not None else None

    async def get_summary_by_content_hash(self, content_hash: str) -> DocumentSummary | None:
        # Slim like list_summaries: the dedup check never touches chunk rows.
        async with _translate_db_outage():
            chunk_counts = _chunk_counts_subquery()
            result = await self._session.execute(
                select(
                    DocumentModel,
                    func.coalesce(chunk_counts.c.chunk_count, 0),
                )
                .outerjoin(chunk_counts, chunk_counts.c.document_id == DocumentModel.id)
                .where(DocumentModel.content_hash == content_hash)
            )
            row = result.first()
        return row_to_summary(row[0], row[1]) if row is not None else None

    async def get_summary_by_id(self, document_id: DocumentId) -> DocumentSummary | None:
        # Slim like the list projection: GET /documents/{id} renders summary
        # fields only, so chunk rows (and their embeddings) are never loaded.
        async with _translate_db_outage():
            chunk_counts = _chunk_counts_subquery()
            result = await self._session.execute(
                select(
                    DocumentModel,
                    func.coalesce(chunk_counts.c.chunk_count, 0),
                )
                .outerjoin(chunk_counts, chunk_counts.c.document_id == DocumentModel.id)
                .where(DocumentModel.id == document_id.value)
            )
            row = result.first()
        return row_to_summary(row[0], row[1]) if row is not None else None

    async def list_summaries(self, limit: int, offset: int) -> list[DocumentSummary]:
        # The slim projection: chunk_count comes from a GROUP BY subquery, so
        # no chunk row (and no 768-float embedding) is ever loaded for lists.
        async with _translate_db_outage():
            chunk_counts = _chunk_counts_subquery()
            result = await self._session.execute(
                select(
                    DocumentModel,
                    func.coalesce(chunk_counts.c.chunk_count, 0),
                )
                .outerjoin(chunk_counts, chunk_counts.c.document_id == DocumentModel.id)
                # Two rows can share a created_at tick; the id tiebreaker makes
                # page boundaries deterministic, or pagination can skip/repeat.
                .order_by(DocumentModel.created_at.desc(), DocumentModel.id)
                .limit(limit)
                .offset(offset)
            )
            rows = result.all()
        return [row_to_summary(model, count) for model, count in rows]

    async def count(self) -> int:
        async with _translate_db_outage():
            result = await self._session.execute(select(func.count(DocumentModel.id)))
            return result.scalar_one()


def _chunk_counts_subquery() -> Subquery:
    """chunk_count per document via GROUP BY — shared by the list projection
    and the slim dedup lookup, so neither hydrates chunk rows."""
    return (
        select(
            ChunkModel.document_id,
            func.count().label("chunk_count"),
        )
        .group_by(ChunkModel.document_id)
        .subquery()
    )


def _is_content_hash_violation(exc: IntegrityError) -> bool:
    """True iff this IntegrityError is the content_hash unique index firing.

    SQLAlchemy's asyncpg adapter drops the structured `constraint_name` when
    translating driver errors, so we walk the cause chain looking for it; the
    final fallback is the message text, which names the index
    (`ix_documents_content_hash`) either way.
    """
    current: BaseException | None = exc
    while current is not None:
        constraint = getattr(current, "constraint_name", None)
        if isinstance(constraint, str):
            return "content_hash" in constraint
        current = current.__cause__
    return "content_hash" in str(exc)
