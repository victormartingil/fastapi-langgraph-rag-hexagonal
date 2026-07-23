"""PydanticAiAnswerGenerator: the LLM adapter.

This is the ONLY module in the project that imports pydantic-ai — by design
(see ADR-0002). The adapter converts our domain question + evidence into a
prompt, runs an Agent whose OUTPUT IS VALIDATED against a Pydantic schema,
and converts the result back into domain objects.

Why structured output matters: we ask the model for JSON shaped like
`AnswerPayload { answer, source_indices }`, and pydantic-ai validates (and
retries on) the model's response. A free-text "the answer is ... (sources:
...)" would need fragile parsing; a validated payload cannot silently drift.

Why INDEX-BASED citations: sources are presented to the model as [1], [2], ...
and the model cites those small integers instead of echoing raw chunk UUIDs.
Small integers are far harder to hallucinate or mangle than 36-character
ids — and out-of-range indices are trivially detectable and dropped.

Resilience: the LLM call is retried with exponential backoff + jitter
(tenacity) — and tenacity is the SINGLE retry authority. The underlying
openai SDK retries 5xx/connection errors internally by default
(`max_retries=2`), which would multiply every adapter attempt into three
HTTP calls with a backoff this adapter neither sees nor controls; the SDK
client is therefore built with `max_retries=0`. If transient failures
outlast the retries, `generate` raises `GenerationUnavailableError` — HTTP
503, symmetric with the retrieval side. Permanent failures (401/403,
exhausted output-validation retries) propagate untouched: they are
configuration/bug signals, not outages, and must stay loud (500-class)
rather than be reported as "temporary".
"""

import httpx
import structlog
from openai import APIConnectionError, AsyncOpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from knowledge_assistant.chat.domain.exceptions import GenerationUnavailableError
from knowledge_assistant.chat.domain.models import Answer, RetrievedChunk, Source
from knowledge_assistant.shared.infrastructure.http_resilience import (
    is_transient_http_error,
)

logger = structlog.get_logger()


def _is_transient_llm_error(exc: BaseException) -> bool:
    """pydantic-ai wraps HTTP failures in its own exception types (not httpx
    exceptions), so the shared policy is extended — locally, keeping the
    vendor imports quarantined in this module (ADR-0002):

    - ModelHTTPError: the server ANSWERED with a status — 5xx and 429 are
      transient, 4xx permanent;
    - ModelAPIError caused by APIConnectionError: the server never answered
      (unreachable, dropped connection) — transient by definition.
    """
    if isinstance(exc, ModelHTTPError):
        return exc.status_code >= 500 or exc.status_code == 429
    if isinstance(exc, ModelAPIError):
        return isinstance(exc.__cause__, APIConnectionError)
    return is_transient_http_error(exc)


SYSTEM_PROMPT = """\
You are a precise knowledge-base assistant. Answer the user's question using
ONLY the provided context chunks. Rules:
- If the context does not contain the answer, say you don't know. NEVER invent
  facts, policies, dates or numbers.
- Cite every claim by listing the NUMBERS of the chunks it comes from
  (chunks are numbered [1], [2], ... in the prompt).
- Keep the answer concise and factual.
"""


class AnswerPayload(BaseModel):
    """The schema the LLM's output is validated against (adapter-local:
    the domain never sees pydantic-ai or this model)."""

    answer: str = Field(description="The answer, grounded in the context chunks")
    source_indices: list[int] = Field(
        description="1-based numbers of the context chunks the answer is based on"
    )


class PydanticAiAnswerGenerator:
    """Implements the `AnswerGenerator` port with pydantic-ai.

    Works against Ollama's OpenAI-compatible endpoint by default (no API key
    needed) and against the real OpenAI API when configured — only base_url,
    api_key and model name differ.
    """

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient,
        max_retries: int = 3,
    ) -> None:
        # The HTTP client is INJECTED, not created here: the composition root
        # owns it (KA_LLM_TIMEOUT_SECONDS becomes its timeout) and closes it
        # on shutdown — an adapter-owned client would leak at process exit.
        self._http_client = http_client
        # `max_retries=0` on the SDK client is deliberate: tenacity (below)
        # is the single retry authority. With the SDK default (2 internal
        # retries), one `max_retries=3` adapter configuration would really be
        # NINE HTTP attempts — and KA_LLM_MAX_RETRIES would lie.
        openai_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            http_client=http_client,
            max_retries=0,
        )
        model = OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(openai_client=openai_client),
        )
        self._agent: Agent[None, AnswerPayload] = Agent(
            model,
            output_type=AnswerPayload,
            system_prompt=SYSTEM_PROMPT,
        )
        self._run_with_retry = retry(
            # Only transient failures (timeouts, connection errors, 5xx) are
            # retried: a 4xx is a configuration problem, not a flaky service.
            retry=retry_if_exception(_is_transient_llm_error),
            stop=stop_after_attempt(max_retries),
            # Exponential backoff + full jitter: without jitter, a fleet of
            # workers that failed together retries in lockstep and re-floods
            # the recovering service (thundering herd).
            wait=wait_exponential(multiplier=0.5, max=8) + wait_random(0, 0.5),
            reraise=True,
        )(self._agent.run)

    @property
    def timeout_seconds(self) -> float:
        """The read timeout of the injected client (exposed so wiring tests
        can prove the composition root passed KA_LLM_TIMEOUT_SECONDS through)."""
        read_timeout = self._http_client.timeout.read
        if read_timeout is None:  # only possible with a custom Timeout object
            msg = "timeout_seconds requires a client built with a scalar timeout"
            raise RuntimeError(msg)
        return read_timeout

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        prompt = self._build_prompt(question, chunks)
        try:
            result = await self._run_with_retry(prompt)
        except Exception as exc:
            # The system's error doctrine, applied to generation: a transient
            # outage that survived every retry is a 503-class domain signal
            # (symmetric with RetrievalUnavailableError on the read side);
            # PERMANENT failures (a dead API key, malformed-output retries
            # exhausted) are configuration/bug signals and stay loud (500).
            # A 200-with-fallback-message would lie twice: "temporary" for a
            # permanent error, and a success shape for a degraded answer.
            if not _is_transient_llm_error(exc):
                raise
            logger.exception("llm_generation_failed", question_length=len(question))
            msg = (
                "The answer-generation service is temporarily unavailable "
                "(LLM unreachable after retries). Please try again shortly."
            )
            raise GenerationUnavailableError(msg) from exc

        payload = result.output
        return Answer(
            text=payload.answer,
            sources=self._resolve_sources(payload.source_indices, chunks),
        )

    @staticmethod
    def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
        context = "\n\n".join(
            f"[{index}] (from {chunk.document_title!r})\n{chunk.content}"
            for index, chunk in enumerate(chunks, start=1)
        )
        return f"Context chunks:\n\n{context}\n\nQuestion: {question}"

    @staticmethod
    def _resolve_sources(
        cited_indices: list[int], chunks: list[RetrievedChunk]
    ) -> tuple[Source, ...]:
        """Map 1-based citation indices back to chunks. Out-of-range indices
        — the index-based equivalent of a hallucinated citation — are dropped,
        not propagated."""
        return tuple(
            Source(
                document_id=chunk.document_id,
                document_title=chunk.document_title,
                chunk_id=chunk.chunk_id,
                excerpt=chunk.content[:300],
                score=chunk.score,
            )
            for index in dict.fromkeys(cited_indices)
            if 1 <= index <= len(chunks)
            for chunk in [chunks[index - 1]]
        )
