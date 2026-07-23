"""HTTP schemas (Pydantic): the API's public contract.

These models exist to describe and validate the HTTP boundary — they are NOT
domain objects and never travel inward. Mappers convert between the two. If
the JSON representation changes, the domain doesn't care.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentResponse(BaseModel):
    """Full representation of an ingested document (or a summary of one:
    the fields coincide — the summary projection is invisible to clients)."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    file_name: str
    created_at: datetime
    chunk_count: int = Field(ge=0)


class DocumentListResponse(BaseModel):
    """One page of documents plus pagination metadata."""

    model_config = ConfigDict(frozen=True)

    documents: list[DocumentResponse]
    total: int = Field(ge=0, description="Total documents in the knowledge base")
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
