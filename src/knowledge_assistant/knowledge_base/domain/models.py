"""Domain models of the knowledge-base context (the write side of the RAG system).

These are PLAIN frozen dataclasses. No SQLAlchemy, no Pydantic, no FastAPI.
That is not purism: it is what makes the core logic trivially testable and
immune to framework upgrades. The mapping to persistence/HTTP representations
happens explicitly in outbound adapters (see `mappers.py` modules).

A `Document` is the aggregate: it owns its `Chunk`s, and chunks are never
persisted independently of their document.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from knowledge_assistant.knowledge_base.domain.exceptions import (
    InvalidDocumentMetadataError,
)
from knowledge_assistant.knowledge_base.domain.value_objects import ChunkText, DocumentId
from knowledge_assistant.shared_kernel.value_objects import EmbeddingVector

MAX_DOCUMENT_TITLE_LENGTH = 200
MAX_FILE_NAME_LENGTH = 255


def validate_document_metadata(title: str, file_name: str) -> None:
    """Enforce limits independently of the HTTP adapter."""
    for field_name, value, max_length in (
        ("title", title, MAX_DOCUMENT_TITLE_LENGTH),
        ("file_name", file_name, MAX_FILE_NAME_LENGTH),
    ):
        if not value.strip():
            raise InvalidDocumentMetadataError(field_name, "cannot be empty")
        if len(value) > max_length:
            raise InvalidDocumentMetadataError(
                field_name, f"must contain at most {max_length} characters"
            )


@dataclass(frozen=True, slots=True)
class Chunk:
    """One embeddable fragment of a Document, with its vector once computed.

    `embedding` is `None` between chunking and embedding — a chunk without a
    vector is a valid intermediate state of the ingestion pipeline.
    """

    id: DocumentId
    text: ChunkText
    position: int
    embedding: EmbeddingVector | None = None


@dataclass(frozen=True, slots=True)
class Document:
    """An ingested document in the permanent knowledge base.

    `content_hash` (SHA-256 of the extracted text) is the deduplication key:
    re-uploading identical content returns the existing document instead of
    creating a twin. `None` only for documents stored before the column
    existed.
    """

    id: DocumentId
    title: str
    file_name: str
    raw_text: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    chunks: tuple[Chunk, ...] = ()
    content_hash: str | None = None

    def __post_init__(self) -> None:
        validate_document_metadata(self.title, self.file_name)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
