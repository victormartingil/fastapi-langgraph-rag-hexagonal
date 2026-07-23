"""The AskQuestion use case: the assistant's public entry point."""

from knowledge_assistant.assistant.application.ports import RagWorkflow
from knowledge_assistant.assistant.domain.exceptions import (
    EmptyQuestionError,
    InvalidQuestionError,
)
from knowledge_assistant.assistant.domain.models import Answer

DEFAULT_TOP_K = 5
MAX_QUESTION_LENGTH = 4_000
MAX_TOP_K = 20


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
        if len(question) > MAX_QUESTION_LENGTH:
            raise InvalidQuestionError(
                f"Question must contain at most {MAX_QUESTION_LENGTH} characters"
            )

        effective_top_k = top_k if top_k is not None else self._default_top_k
        if not 1 <= effective_top_k <= MAX_TOP_K:
            raise InvalidQuestionError(f"top_k must be between 1 and {MAX_TOP_K}")
        return await self._workflow.run(question, effective_top_k)
