"""HTTP schemas of the assistant context: the read side's public contract."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Question = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_000),
]


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: Question = Field(examples=["Can I return a product after two months?"])
    # Omit to use the server default (KA_RETRIEVAL_TOP_K).
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    document_title: str
    chunk_id: str
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    sources: list[SourceResponse]
