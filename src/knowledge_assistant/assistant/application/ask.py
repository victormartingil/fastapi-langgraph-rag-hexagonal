"""The AskQuestion use case: the read side's single entry point.

It validates the input, runs the LangGraph pipeline, and unwraps the answer.
The graph is injected as a compiled LangGraph `CompiledStateGraph`; the use
case knows the SHAPE of the state (`RagState`) but nothing about retrieval
dialects or LLM vendors.
"""

from typing import cast

from langgraph.graph.state import CompiledStateGraph

from knowledge_assistant.assistant.application.graph.state import RagState
from knowledge_assistant.assistant.domain.exceptions import EmptyQuestionError
from knowledge_assistant.assistant.domain.models import Answer

DEFAULT_TOP_K = 5


class AskQuestion:
    """Answer a natural-language question with cited sources (or refuse).

    `default_top_k` is the server-side retrieval breadth (wired from
    `KA_RETRIEVAL_TOP_K` by the composition root); a caller-provided `top_k`
    always wins.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[RagState, None, RagState, RagState],
        *,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._graph = graph
        self._default_top_k = default_top_k

    async def execute(self, question: str, top_k: int | None = None) -> Answer:
        if not question.strip():
            raise EmptyQuestionError("Question cannot be empty")

        effective_top_k = top_k if top_k is not None else self._default_top_k
        initial_state: RagState = {"question": question, "top_k": effective_top_k}
        final_state = cast("RagState", await self._graph.ainvoke(initial_state))
        return final_state["answer"]
