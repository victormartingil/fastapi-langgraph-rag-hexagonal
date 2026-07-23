"""Pure decisions shared by RAG workflow implementations."""

from typing import Literal

from knowledge_assistant.assistant.domain.models import Answer, RetrievedChunk

REFUSAL_MESSAGE = (
    "I could not find any relevant information in the knowledge base to answer "
    "that question. Please ingest relevant documents first, or rephrase the question."
)


def filter_relevant_evidence(
    chunks: list[RetrievedChunk], min_relevance_score: float
) -> list[RetrievedChunk]:
    """Keep evidence that clears the configured relevance threshold."""
    return [chunk for chunk in chunks if chunk.score >= min_relevance_score]


def decide_answer_route(
    relevant_chunks: list[RetrievedChunk],
) -> Literal["generate", "refuse"]:
    """Generate only when evidence exists."""
    return "generate" if relevant_chunks else "refuse"


def refusal_answer() -> Answer:
    """Build the deterministic answer for missing evidence."""
    return Answer(text=REFUSAL_MESSAGE, sources=(), is_refusal=True)
