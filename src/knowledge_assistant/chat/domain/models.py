"""Domain models of the chat context (the read side of the RAG system).

Note what is NOT here: no LangGraph types, no Pydantic-AI types, no SQL. The
graph orchestrates these plain objects; the LLM adapter maps them to whatever
structured-output schema the vendor SDK needs.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk fetched by the retriever, with the score that ranked it."""

    chunk_id: str
    document_id: str
    document_title: str
    content: str
    score: float


@dataclass(frozen=True)
class Source:
    """A citation attached to an answer: where did this claim come from?"""

    document_id: str
    document_title: str
    chunk_id: str
    excerpt: str
    score: float


@dataclass(frozen=True)
class Answer:
    """The final product of the Q&A pipeline.

    `sources` is part of the domain model — not an afterthought — because an
    answer without provenance is not acceptable in this system.
    """

    text: str
    sources: tuple[Source, ...]
