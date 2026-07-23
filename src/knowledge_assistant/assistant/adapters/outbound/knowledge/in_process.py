"""In-process adapter from the assistant port to the knowledge-base API."""

from knowledge_assistant.assistant.domain.exceptions import RetrievalUnavailableError
from knowledge_assistant.assistant.domain.models import RetrievedChunk
from knowledge_assistant.knowledge_base.application.exceptions import (
    KnowledgeBaseUnavailableError,
)
from knowledge_assistant.knowledge_base.application.queries import SearchKnowledge


class InProcessKnowledgeSearchAdapter:
    """Translate the public knowledge-base use case into assistant concepts.

    These are the only sanctioned cross-context imports, and both target the
    public knowledge-base application boundary. Replacing this bridge with an
    HTTP or messaging adapter leaves the assistant application unchanged.
    """

    def __init__(self, search_knowledge: SearchKnowledge) -> None:
        self._search_knowledge = search_knowledge

    async def search(self, question: str, limit: int) -> list[RetrievedChunk]:
        try:
            hits = await self._search_knowledge.execute(question, limit)
        except KnowledgeBaseUnavailableError as exc:
            raise RetrievalUnavailableError(str(exc)) from exc
        return [
            RetrievedChunk(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                content=hit.content,
                score=hit.score,
            )
            for hit in hits
        ]
