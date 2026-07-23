"""Contract tests for the explicit in-process bounded-context bridge."""

import pytest

from knowledge_assistant.assistant.adapters.outbound.knowledge.in_process import (
    InProcessKnowledgeSearchAdapter,
)
from knowledge_assistant.assistant.domain.exceptions import RetrievalUnavailableError
from knowledge_assistant.knowledge_base.application.queries import SearchKnowledge
from knowledge_assistant.knowledge_base.application.read_models import KnowledgeHit
from knowledge_assistant.knowledge_base.domain.exceptions import (
    KnowledgeBaseUnavailableError,
)


class StubKnowledgeRetriever:
    def __init__(
        self,
        hits: list[KnowledgeHit] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.hits = hits or []
        self.error = error

    async def retrieve(self, question: str, limit: int) -> list[KnowledgeHit]:
        if self.error is not None:
            raise self.error
        return self.hits[:limit]


async def test_bridge_maps_the_public_knowledge_projection() -> None:
    hit = KnowledgeHit("chunk-1", "doc-1", "Policy", "Grounded fact", 0.42)
    adapter = InProcessKnowledgeSearchAdapter(SearchKnowledge(StubKnowledgeRetriever([hit])))

    [chunk] = await adapter.search("question", 1)

    assert chunk.chunk_id == hit.chunk_id
    assert chunk.document_title == hit.document_title
    assert chunk.score == hit.score


async def test_bridge_translates_context_specific_outage_signal() -> None:
    search = SearchKnowledge(StubKnowledgeRetriever(error=KnowledgeBaseUnavailableError()))
    adapter = InProcessKnowledgeSearchAdapter(search)

    with pytest.raises(RetrievalUnavailableError, match="temporarily unavailable"):
        await adapter.search("question", 5)
