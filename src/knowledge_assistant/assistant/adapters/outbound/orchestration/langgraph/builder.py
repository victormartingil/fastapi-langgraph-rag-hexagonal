"""Assembly of the RAG graph.

The graph is the READ-SIDE use case expressed as a state machine:

    START → retrieve → grade ──┬─ (evidence found)   → generate → END
                               └─ (no evidence)      → refuse   → END

LangGraph is an outbound orchestration adapter. The decisions it coordinates
live as pure application policies, while this module owns graph-specific
state, nodes, edges, and compilation.
"""

from typing import cast

from langgraph.graph import END, START, StateGraph

from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph import nodes
from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph.state import RagState
from knowledge_assistant.assistant.application.ports import AnswerGenerator, KnowledgeSearch
from knowledge_assistant.assistant.domain.models import Answer


def build_rag_graph(
    knowledge_search: KnowledgeSearch,
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
    graph.add_node("retrieve", nodes.make_retrieve_node(knowledge_search))  # type: ignore[call-overload]
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


class LangGraphRagWorkflow:
    """LangGraph implementation of the application ``RagWorkflow`` port."""

    def __init__(
        self,
        knowledge_search: KnowledgeSearch,
        answer_generator: AnswerGenerator,
        *,
        min_relevance_score: float,
    ) -> None:
        self._graph = build_rag_graph(
            knowledge_search,
            answer_generator,
            min_relevance_score=min_relevance_score,
        ).compile()

    async def run(self, question: str, top_k: int) -> Answer:
        initial_state: RagState = {"question": question, "top_k": top_k}
        final_state = cast("RagState", await self._graph.ainvoke(initial_state))
        return final_state["answer"]
