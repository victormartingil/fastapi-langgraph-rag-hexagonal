"""Nodes of the RAG graph.

Each node is a small async function: state in, partial state update out.
Nodes are created by FACTORY FUNCTIONS that close over the ports they need —
this is how dependencies reach the graph without LangGraph knowing anything
about DI.

Crucially, nodes call PORTS (`KnowledgeSearch`, `AnswerGenerator`), never
concrete adapters. Unit tests exercise every node with hand-written fakes and
zero infrastructure.
"""

from collections.abc import Awaitable, Callable

from knowledge_assistant.assistant.application.graph.state import RagState
from knowledge_assistant.assistant.application.ports import AnswerGenerator, KnowledgeSearch
from knowledge_assistant.assistant.domain.models import Answer

Node = Callable[[RagState], Awaitable[dict[str, object]]]

REFUSAL_MESSAGE = (
    "I could not find any relevant information in the knowledge base to answer "
    "that question. Please ingest relevant documents first, or rephrase the question."
)


def make_retrieve_node(knowledge_search: KnowledgeSearch) -> Node:
    """Node 1 — fetch candidate chunks from the knowledge base."""

    async def retrieve(state: RagState) -> dict[str, object]:
        chunks = await knowledge_search.search(state["question"], limit=state["top_k"])
        return {"retrieved_chunks": chunks}

    return retrieve


def make_grade_node(min_relevance_score: float) -> Node:
    """Node 2 — keep only chunks whose score clears the relevance bar.

    Grading here is a deterministic threshold on the hybrid RRF score: simple,
    fast, and fully unit-testable. An LLM-as-judge grader is a drop-in
    replacement (one more port call) deferred to the roadmap.
    """

    async def grade(state: RagState) -> dict[str, object]:
        retrieved = state.get("retrieved_chunks", [])
        relevant = [chunk for chunk in retrieved if chunk.score >= min_relevance_score]
        return {"relevant_chunks": relevant}

    return grade


def route_after_grading(state: RagState) -> str:
    """Conditional edge: no relevant evidence → honest refusal, never hallucinate."""
    return "generate" if state.get("relevant_chunks") else "refuse"


def make_generate_node(generator: AnswerGenerator) -> Node:
    """Node 3a — answer grounded in the surviving evidence."""

    async def generate(state: RagState) -> dict[str, object]:
        answer = await generator.generate(
            state["question"],
            chunks=state.get("relevant_chunks", []),
        )
        return {"answer": answer}

    return generate


def make_refuse_node() -> Node:
    """Node 3b — the honest refusal path.

    This node never calls an LLM: with zero relevant context there is nothing
    to ground an answer on, so the correct output is a fixed, truthful refusal.
    """

    async def refuse(state: RagState) -> dict[str, object]:
        return {"answer": Answer(text=REFUSAL_MESSAGE, sources=())}

    return refuse
