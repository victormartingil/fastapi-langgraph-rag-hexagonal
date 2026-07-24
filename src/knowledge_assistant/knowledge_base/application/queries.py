"""Read use cases exposed by the knowledge-base context."""

from knowledge_assistant.knowledge_base.application.ports import (
    DocumentRepository,
    OpenKnowledgeRetriever,
    RetrievalStrategy,
)
from knowledge_assistant.knowledge_base.application.read_models import (
    DocumentPage,
    DocumentSummary,
    KnowledgeHit,
)
from knowledge_assistant.knowledge_base.domain.exceptions import DocumentNotFoundError
from knowledge_assistant.knowledge_base.domain.value_objects import DocumentId


class GetDocument:
    """Fetch a slim document projection or report that it does not exist."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, document_id: DocumentId) -> DocumentSummary:
        summary = await self._repository.get_summary_by_id(document_id)
        if summary is None:
            raise DocumentNotFoundError(str(document_id))
        return summary


class ListDocuments:
    """List document summaries page by page."""

    def __init__(self, repository: DocumentRepository) -> None:
        self._repository = repository

    async def execute(self, limit: int, offset: int) -> DocumentPage:
        summaries = await self._repository.list_summaries(limit, offset)
        total = await self._repository.count()
        return DocumentPage(items=tuple(summaries), total=total, limit=limit, offset=offset)


class SearchKnowledge:
    """Public application API for ranked knowledge search."""

    def __init__(self, open_retriever: OpenKnowledgeRetriever) -> None:
        self._open_retriever = open_retriever

    async def execute(
        self,
        question: str,
        limit: int,
        *,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
    ) -> list[KnowledgeHit]:
        async with self._open_retriever() as retriever:
            return await retriever.retrieve(question, limit, strategy=strategy)
