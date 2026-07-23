"""Explicit mappers: domain answer -> HTTP schema.

Symmetric with the knowledge-base context: nothing crosses the HTTP boundary
without a named, reviewable function.
"""

from knowledge_assistant.assistant.adapters.inbound.http.schemas import (
    ChatResponse,
    SourceResponse,
)
from knowledge_assistant.assistant.domain.models import Answer


def answer_to_response(answer: Answer) -> ChatResponse:
    return ChatResponse(
        answer=answer.text,
        sources=[
            SourceResponse(
                document_id=source.document_id,
                document_title=source.document_title,
                chunk_id=source.chunk_id,
                excerpt=source.excerpt,
                score=source.score,
            )
            for source in answer.sources
        ],
    )
