"""State of the RAG graph.

LangGraph passes ONE state object through the graph; each node returns a
partial update that is merged into it. We model it as a `TypedDict` — the
idiomatic LangGraph 1.x choice — carrying plain domain objects, so the state
itself stays free of vendor types.
"""

from typing import TypedDict

from knowledge_assistant.chat.domain.models import Answer, RetrievedChunk


class RagState(TypedDict, total=False):
    """State flowing through retrieve → grade → (generate | refuse).

    `total=False` because nodes enrich the state progressively: at START only
    `question` and `top_k` exist.
    """

    question: str
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    answer: Answer
