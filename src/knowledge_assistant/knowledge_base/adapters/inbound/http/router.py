"""HTTP adapter of the knowledge-base context: the write side's API.

Routers are THIN: parse the HTTP request, call the use case, map the result.
No business rules here. Use cases arrive via FastAPI's `Depends`, which calls
provider functions from the composition root (`bootstrap.py`) — that is the
only place where concrete adapters are chosen.

The whole router is behind the optional API-key guard (`require_api_key`);
when `KA_API_KEY` is unset the guard is a no-op.
"""

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)

from knowledge_assistant import bootstrap
from knowledge_assistant.bootstrap import Container
from knowledge_assistant.knowledge_base.adapters.inbound.http.mappers import (
    document_to_response,
    summary_to_response,
)
from knowledge_assistant.knowledge_base.adapters.inbound.http.schemas import (
    DocumentListResponse,
    DocumentResponse,
)
from knowledge_assistant.knowledge_base.application.ingest import IngestDocument
from knowledge_assistant.knowledge_base.application.queries import (
    GetDocument,
    ListDocuments,
)
from knowledge_assistant.knowledge_base.domain.models import Document
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
    dependencies=[Depends(bootstrap.require_api_key)],
)

# Read one extra byte beyond the limit to detect overflow without buffering
# the whole upload: a 10 MB cap never costs more than 10 MB + 1 B of memory.
OVERFLOW_PROBE_BYTES = 1


@router.post("", status_code=status.HTTP_201_CREATED, response_model=DocumentResponse)
async def ingest_document(
    file: Annotated[UploadFile, File(description="A .md, .txt or .pdf document")],
    use_case: Annotated[IngestDocument, Depends(bootstrap.provide_ingest_document)],
    app_container: Annotated[Container, Depends(bootstrap.get_container)],
    response: Response,
    title: Annotated[str | None, Form(description="Optional human title")] = None,
) -> DocumentResponse:
    """Ingest a document. Idempotent by content: re-uploading identical
    content returns the EXISTING document with 200 (instead of a 201 twin).

    Dedup keys on CONTENT (SHA-256 of the extracted text), not on file name
    or title: a re-upload of the same content under a new title keeps the
    ORIGINAL document's title. Two concurrent identical uploads race safely —
    the unique index picks a winner and the loser gets the same 200 answer.
    """
    max_bytes = int(app_container.settings.max_upload_size_mb * 1024 * 1024)
    data = await file.read(max_bytes + OVERFLOW_PROBE_BYTES)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Upload exceeds the {app_container.settings.max_upload_size_mb:g} MB limit",
        )

    result = await use_case.execute(file_name=file.filename or "untitled", data=data, title=title)
    if not result.created:
        response.status_code = status.HTTP_200_OK
    # Duplicates come back as a slim summary (no chunk hydration); new
    # documents as the full aggregate. Both map to the same response schema.
    if isinstance(result.document, Document):
        return document_to_response(result.document)
    return summary_to_response(result.document)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    use_case: Annotated[ListDocuments, Depends(bootstrap.provide_list_documents)],
    limit: Annotated[int, Query(ge=1, le=100, description="Page size")] = 20,
    offset: Annotated[int, Query(ge=0, description="Items to skip")] = 0,
) -> DocumentListResponse:
    page = await use_case.execute(limit=limit, offset=offset)
    return DocumentListResponse(
        documents=[summary_to_response(summary) for summary in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,  # FastAPI validates the UUID format: malformed → 422
    use_case: Annotated[GetDocument, Depends(bootstrap.provide_get_document)],
) -> DocumentResponse:
    summary = await use_case.execute(DocumentId(document_id))
    return summary_to_response(summary)
