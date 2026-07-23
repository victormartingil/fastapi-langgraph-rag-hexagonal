"""Output ports of the chat context.

Two seams are enough for the whole read side:

- `ChunkRetriever`: "give me the chunks most relevant to this question".
  The PostgreSQL+pgvector hybrid implementation is ONE possible adapter.
- `AnswerGenerator`: "turn question + evidence into a cited answer".
  The Pydantic-AI implementation is ONE possible adapter.

The LangGraph pipeline calls these protocols and has no idea that Postgres,
Ollama or OpenAI exist.
"""

from typing import Protocol

from knowledge_assistant.chat.domain.models import Answer, RetrievedChunk


class ChunkRetriever(Protocol):
    """Port toward the knowledge base search engine."""

    async def retrieve(self, question: str, limit: int) -> list[RetrievedChunk]:
        """Return up to `limit` chunks ranked by relevance to `question`."""
        ...


class AnswerGenerator(Protocol):
    """Port toward the LLM that writes the final, cited answer."""

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        """Generate an answer grounded ONLY in `chunks`, citing its sources.

        Contract: a transient outage that outlives the adapter's retries
        surfaces as `GenerationUnavailableError` (-> HTTP 503); permanent
        failures propagate as-is (500-class). Adapters never fabricate a
        degraded 200.
        """
        ...
