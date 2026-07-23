"""HTTP adapter of the chat context: POST /api/v1/chat.

Thin as every router: validate → call use case → map domain answer to the
response schema. The LangGraph pipeline is entirely hidden behind the
`AskQuestion` use case.
"""

from typing import Annotated

from fastapi import APIRouter, Depends

from knowledge_assistant import container
from knowledge_assistant.chat.application.service import AskQuestion
from knowledge_assistant.chat.infrastructure.http.mappers import answer_to_response
from knowledge_assistant.chat.infrastructure.http.schemas import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["chat"],
    dependencies=[Depends(container.require_api_key)],
)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    use_case: Annotated[AskQuestion, Depends(container.provide_ask_question)],
) -> ChatResponse:
    answer = await use_case.execute(request.question, top_k=request.top_k)
    return answer_to_response(answer)
