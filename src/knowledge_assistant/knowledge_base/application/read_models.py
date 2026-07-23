"""Read models (projections) of the knowledge-base context.

A read model is NOT the aggregate: `DocumentSummary` exists because the list
endpoint needs `chunk_count` without hydrating every chunk (and its
768-float embedding) from the database. Keeping it in the APPLICATION layer —
not the domain — says exactly what it is: a query-side projection shaped by a
use case, not a business concept.

CQRS in miniature: commands go through the aggregate (`Document`), queries
through projections (`DocumentSummary`).
"""

from dataclasses import dataclass
from datetime import datetime

from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    """What the list endpoint knows about a document — no chunks attached."""

    id: DocumentId
    title: str
    file_name: str
    created_at: datetime
    chunk_count: int


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of summaries plus the total, so clients can paginate."""

    items: tuple[DocumentSummary, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    """A ranked search result exposed by the knowledge-base application API.

    This projection deliberately contains no persistence model or vector. It
    is the stable boundary consumed by in-process or remote adapters.
    """

    chunk_id: str
    document_id: str
    document_title: str
    content: str
    score: float
