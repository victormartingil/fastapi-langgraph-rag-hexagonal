"""Assembly of the RAG graph.

The graph is the READ-SIDE use case expressed as a state machine:

    START → retrieve → grade ──┬─ (evidence found)   → generate → END
                               └─ (no evidence)      → refuse   → END

Why does this live in the APPLICATION layer? Because "retrieve, grade, then
answer or refuse" is application policy — the same policy would exist with a
different orchestrator. LangGraph is used as a library here; the domain does
not import it (see ADR-0002 and the import-linter contracts).
"""

from langgraph.graph import END, START, StateGraph

from knowledge_assistant.chat.application.graph import nodes
from knowledge_assistant.chat.application.graph.state import RagState
from knowledge_assistant.chat.application.ports import AnswerGenerator, ChunkRetriever


def build_rag_graph(
    retriever: ChunkRetriever,
    answer_generator: AnswerGenerator,
    *,
    min_relevance_score: float,
) -> StateGraph[RagState, None, RagState, RagState]:
    """Wire the nodes and edges; returns an UNcompiled graph.

    Callers compile it (`.compile()`), which keeps the wiring testable and
    leaves room for adding a checkpointer later without touching this code.
    """
    graph = StateGraph(RagState)

    # NOTE for mypy users: LangGraph 1.x types `add_node` with a `NodeInputT`
    # bound to a `StateLike` Protocol, which TypedDict instance types do not
    # satisfy structurally under `mypy --strict` (upstream typing limitation).
    # The runtime contract is exactly what our node factories produce:
    # `(RagState) -> partial state update`. Hence the targeted ignores.
    graph.add_node("retrieve", nodes.make_retrieve_node(retriever))  # type: ignore[call-overload]
    graph.add_node("grade", nodes.make_grade_node(min_relevance_score))  # type: ignore[call-overload]
    graph.add_node("generate", nodes.make_generate_node(answer_generator))  # type: ignore[call-overload]
    graph.add_node("refuse", nodes.make_refuse_node())  # type: ignore[call-overload]

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        nodes.route_after_grading,
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph
