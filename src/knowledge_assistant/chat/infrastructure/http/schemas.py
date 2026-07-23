"""HTTP schemas of the chat context: the read side's public contract."""

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1, examples=["Can I return a product after two months?"])
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
