"""Output ports of the knowledge-base context.

A port is a `typing.Protocol`: a STRUCTURAL interface. Anything with matching
methods satisfies it — no inheritance required (the Pythonic answer to Java
interfaces). Use cases depend on these protocols; outbound adapters
implement them; tests replace them with hand-written fakes.

Ports are small and focused (Interface Segregation): the use case that ingests
documents does not know what a database is.
"""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol

from knowledge_assistant.knowledge_base.application.read_models import (
    DocumentSummary,
    KnowledgeHit,
)
from knowledge_assistant.knowledge_base.domain.models import Document
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId
from knowledge_assistant.shared_kernel.value_objects import EmbeddingVector

__all__ = [
    "DocumentRepository",
    "EmbeddingProvider",
    "KnowledgeRetriever",
    "OpenRepository",
    "TextExtractor",
]


class EmbeddingProvider(Protocol):
    """Port toward a model that turns text into dense vectors."""

    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        """Embed a batch while preserving input order."""
        ...


class DocumentRepository(Protocol):
    """Persistence port for the Document aggregate (plus query projections).

    Commands go through the aggregate (`save`, `get_by_id`); the list query
    goes through the slim projection (`list_summaries`) so rendering
    `chunk_count` never hydrates chunks or their embeddings.
    """

    async def save(self, document: Document) -> None:
        """Persist a document together with all of its chunks."""
        ...

    async def get_by_id(self, document_id: DocumentId) -> Document | None:
        """Return the document (with chunks) or None if it does not exist."""
        ...

    async def get_summary_by_content_hash(self, content_hash: str) -> DocumentSummary | None:
        """SLIM deduplication lookup: the summary of the document with this
        content hash, if any. Deliberately not the full aggregate — the dedup
        check and the race recovery only need to SAY "already exists", so
        chunks (and their 768-float embeddings) are never hydrated."""
        ...

    async def get_summary_by_id(self, document_id: DocumentId) -> DocumentSummary | None:
        """SLIM single-document lookup for GET /documents/{id}: the response
        schema only exposes summary fields, so hydrating the full aggregate
        (raw_text + embeddings) just to drop it would be waste."""
        ...

    async def list_summaries(self, limit: int, offset: int) -> list[DocumentSummary]:
        """One page of summaries (chunk counts included, chunks NOT loaded)."""
        ...

    async def count(self) -> int:
        """Total number of documents (for pagination metadata)."""
        ...


# A factory that opens a repository around a SHORT-LIVED unit of work: each
# call returns a fresh async context manager; entering it acquires a session
# (and its transaction), leaving it commits or rolls back and releases the
# connection. `IngestDocument` depends on this — never on a long-lived
# repository — so slow work (extraction, embedding) never pins a pooled
# database connection (ADR-0005).
OpenRepository = Callable[[], AbstractAsyncContextManager[DocumentRepository]]


class TextExtractor(Protocol):
    """Port for turning uploaded bytes into plain text.

    Adapters declare which files they handle via `supports`, and the use case
    picks the first capable one — a tiny Strategy/Chain-of-Responsibility that
    keeps `IngestDocument` independent of file formats.
    """

    def supports(self, file_name: str) -> bool:
        """Return True if this extractor can handle `file_name`."""
        ...

    def extract(self, file_name: str, data: bytes) -> str:
        """Extract plain text from raw file bytes."""
        ...


class KnowledgeRetriever(Protocol):
    """Port for ranked search over the knowledge base."""

    async def retrieve(self, question: str, limit: int) -> list[KnowledgeHit]:
        """Return up to ``limit`` hits ranked by relevance."""
        ...
