"""Explicit mappers: domain model / read model -> HTTP schema.

(The other direction is unnecessary here: ingestion input arrives as raw file
bytes + form fields, not as a JSON body needing mapping.)
"""

from knowledge_assistant.documents.application.read_models import DocumentSummary
from knowledge_assistant.documents.domain.models import Document
from knowledge_assistant.documents.infrastructure.http.schemas import DocumentResponse


def document_to_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=str(document.id),
        title=document.title,
        file_name=document.file_name,
        created_at=document.created_at,
        chunk_count=document.chunk_count,
    )


def summary_to_response(summary: DocumentSummary) -> DocumentResponse:
    return DocumentResponse(
        id=str(summary.id),
        title=summary.title,
        file_name=summary.file_name,
        created_at=summary.created_at,
        chunk_count=summary.chunk_count,
    )
