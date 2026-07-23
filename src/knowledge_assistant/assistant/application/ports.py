"""Output ports of the assistant context.

Two seams are enough for the whole read side:

- `KnowledgeSearch`: "give me the evidence most relevant to this question".
  An in-process bridge to the knowledge-base context is ONE possible adapter.
- `AnswerGenerator`: "turn question + evidence into a cited answer".
  The Pydantic-AI implementation is ONE possible adapter.

The LangGraph pipeline calls these protocols and has no idea that Postgres,
Ollama or OpenAI exist.
"""

from typing import Protocol

from knowledge_assistant.assistant.domain.models import Answer, RetrievedChunk


class KnowledgeSearch(Protocol):
    """Port toward a searchable source of knowledge."""

    async def search(self, question: str, limit: int) -> list[RetrievedChunk]:
        """Return up to ``limit`` evidence chunks ranked by relevance."""
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
