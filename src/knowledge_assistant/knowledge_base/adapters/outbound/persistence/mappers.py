"""Explicit mappers: persistence model <-> domain model.

No framework "automagic" mapping: the conversion is plain, boring, reviewable
functions. This is where the impedance mismatch between an ORM row and a
frozen domain dataclass is absorbed — once, in one place.
"""

from knowledge_assistant.knowledge_base.adapters.outbound.persistence.models import (
    ChunkModel,
    DocumentModel,
)
from knowledge_assistant.knowledge_base.application.read_models import DocumentSummary
from knowledge_assistant.knowledge_base.domain.models import Chunk, Document
from knowledge_assistant.knowledge_base.domain.value_objects import ChunkText, DocumentId
from knowledge_assistant.shared_kernel.value_objects import EmbeddingVector


def chunk_to_model(chunk: Chunk, document_id: DocumentId) -> ChunkModel:
    """Domain -> ORM. Requires the embedding to be present: a chunk without a
    vector is an intermediate state and must never reach the database."""
    if chunk.embedding is None:
        msg = "Cannot persist a chunk without an embedding"
        raise ValueError(msg)
    return ChunkModel(
        id=chunk.id.value,
        document_id=document_id.value,
        position=chunk.position,
        content=str(chunk.text),
        embedding=list(chunk.embedding.values),
    )


def chunk_to_domain(model: ChunkModel) -> Chunk:
    """ORM -> domain."""
    return Chunk(
        id=DocumentId(model.id),
        text=ChunkText(model.content),
        position=model.position,
        embedding=EmbeddingVector(tuple(float(v) for v in model.embedding)),
    )


def document_to_model(document: Document) -> DocumentModel:
    """Domain -> ORM, including the owned chunks (aggregate persistence)."""
    return DocumentModel(
        id=document.id.value,
        title=document.title,
        file_name=document.file_name,
        raw_text=document.raw_text,
        created_at=document.created_at,
        content_hash=document.content_hash,
        chunks=[chunk_to_model(chunk, document.id) for chunk in document.chunks],
    )


def document_to_domain(model: DocumentModel) -> Document:
    """ORM -> domain. Expects `chunks` to be loaded (selectin)."""
    return Document(
        id=DocumentId(model.id),
        title=model.title,
        file_name=model.file_name,
        raw_text=model.raw_text,
        created_at=model.created_at,
        chunks=tuple(chunk_to_domain(chunk) for chunk in model.chunks),
        content_hash=model.content_hash,
    )


def row_to_summary(model: DocumentModel, chunk_count: int) -> DocumentSummary:
    """ORM row (+ count subquery value) -> read model. No chunks involved —
    that is the entire point of the summary projection."""
    return DocumentSummary(
        id=DocumentId(model.id),
        title=model.title,
        file_name=model.file_name,
        created_at=model.created_at,
        chunk_count=chunk_count,
    )
