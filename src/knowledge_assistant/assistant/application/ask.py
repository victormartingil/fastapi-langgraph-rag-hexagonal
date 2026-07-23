"""The AskQuestion use case: the assistant's public entry point."""

from knowledge_assistant.assistant.application.ports import RagWorkflow
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
        workflow: RagWorkflow,
        *,
        default_top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._workflow = workflow
        self._default_top_k = default_top_k

    async def execute(self, question: str, top_k: int | None = None) -> Answer:
        if not question.strip():
            raise EmptyQuestionError("Question cannot be empty")

        effective_top_k = top_k if top_k is not None else self._default_top_k
        return await self._workflow.run(question, effective_top_k)
