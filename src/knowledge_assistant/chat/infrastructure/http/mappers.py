"""Explicit mappers: domain answer -> HTTP schema.

Symmetric with the documents context: nothing crosses the HTTP boundary
without a named, reviewable function.
"""

from knowledge_assistant.chat.domain.models import Answer
from knowledge_assistant.chat.infrastructure.http.schemas import (
    ChatResponse,
    SourceResponse,
)


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
