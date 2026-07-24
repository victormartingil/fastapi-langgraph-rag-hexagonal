"""Unit tests for the RAG graph: nodes, routing, and the full compiled flow.

The graph is tested against fake ports — LangGraph itself needs no Docker and
no LLM — including the most important behavior of the whole system: when no
relevant context survives grading, the answer is an honest refusal.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from knowledge_assistant.assistant.adapters.outbound.knowledge.in_process import (
    InProcessKnowledgeSearchAdapter,
)
from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph.builder import (
    LangGraphRagWorkflow,
)
from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph.nodes import (
    make_generate_node,
    make_grade_node,
    make_retrieve_node,
    route_after_grading,
)
from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph.state import RagState
from knowledge_assistant.assistant.application.ask import AskQuestion
from knowledge_assistant.assistant.application.policies import REFUSAL_MESSAGE
from knowledge_assistant.assistant.application.ports import KnowledgeSearch
from knowledge_assistant.assistant.domain.exceptions import (
    EmptyQuestionError,
    InvalidQuestionError,
    RetrievalUnavailableError,
)
from knowledge_assistant.assistant.domain.models import Answer, RetrievedChunk, Source
from knowledge_assistant.knowledge_base.application.queries import SearchKnowledge
from knowledge_assistant.knowledge_base.application.read_models import KnowledgeHit
from knowledge_assistant.knowledge_base.application.retrieval import RetrievalStrategy
from tests.unit.fakes import (
    FailingKnowledgeSearch,
    FakeAnswerGenerator,
    FakeKnowledgeSearch,
    make_retrieved_chunk,
)


def make_answer(chunks_count: int = 1) -> Answer:
    return Answer(
        text="You can return it within 30 days.",
        sources=tuple(
            Source(
                document_id="doc-1",
                document_title="Return Policy",
                chunk_id=f"chunk-{i}",
                excerpt="...",
                score=0.05,
            )
            for i in range(chunks_count)
        ),
    )


def test_affirmative_answer_requires_at_least_one_source() -> None:
    with pytest.raises(ValueError, match="at least one source"):
        Answer(text="Unsupported claim", sources=())


def test_answer_invariants_reject_blank_text_and_sourced_refusal() -> None:
    source = make_answer().sources[0]
    with pytest.raises(ValueError, match="cannot be empty"):
        Answer(text=" ", sources=(source,))
    with pytest.raises(ValueError, match="refusal"):
        Answer(text="No answer", sources=(source,), is_refusal=True)


class TestNodes:
    async def test_retrieve_node_calls_the_retriever_port(self) -> None:
        retriever = FakeKnowledgeSearch([make_retrieved_chunk()])
        node = make_retrieve_node(retriever)

        update = await node(RagState(question="q?", top_k=5))

        assert update["retrieved_chunks"] == retriever.chunks
        assert retriever.calls == [("q?", 5)]

    async def test_grade_node_filters_below_the_threshold(self) -> None:
        node = make_grade_node(min_relevance_score=0.028)
        state = RagState(
            retrieved_chunks=[
                make_retrieved_chunk(chunk_id="strong", score=0.033),
                make_retrieved_chunk(chunk_id="weak", score=0.016),
            ]
        )

        update = await node(state)

        relevant = update["relevant_chunks"]
        assert isinstance(relevant, list)
        assert [c.chunk_id for c in relevant] == ["strong"]

    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [([make_retrieved_chunk()], "generate"), ([], "refuse")],
        ids=["evidence-found", "no-evidence"],
    )
    def test_route_after_grading(self, chunks: list[RetrievedChunk], expected: str) -> None:
        assert route_after_grading(RagState(relevant_chunks=chunks)) == expected

    async def test_generate_node_calls_the_generator_port(self) -> None:
        generator = FakeAnswerGenerator(make_answer())
        chunks = [make_retrieved_chunk()]
        node = make_generate_node(generator)

        update = await node(RagState(question="q?", relevant_chunks=chunks))

        assert update["answer"] == generator.answer
        assert generator.calls == [("q?", chunks)]


class TestAskQuestionOverCompiledGraph:
    def build_use_case(
        self, retriever: KnowledgeSearch, generator: FakeAnswerGenerator
    ) -> AskQuestion:
        workflow = LangGraphRagWorkflow(retriever, generator, min_relevance_score=0.028)
        return AskQuestion(workflow)

    async def test_relevant_evidence_produces_a_cited_answer(self) -> None:
        retriever = FakeKnowledgeSearch([make_retrieved_chunk(score=0.033)])
        generator = FakeAnswerGenerator(make_answer())

        answer = await self.build_use_case(retriever, generator).execute(
            "Can I return a product after two months?"
        )

        assert answer.text == "You can return it within 30 days."
        assert len(answer.sources) == 1
        assert len(generator.calls) == 1  # the LLM was actually asked

    async def test_retrieval_scope_is_closed_before_generation(self) -> None:
        class ScopeRecorder:
            currently_open = 0

            @asynccontextmanager
            async def open_retriever(self) -> AsyncIterator["ScopeRecorder"]:
                self.currently_open += 1
                try:
                    yield self
                finally:
                    self.currently_open -= 1

            async def retrieve(
                self,
                question: str,
                limit: int,
                *,
                strategy: RetrievalStrategy = RetrievalStrategy.HYBRID,
            ) -> list[KnowledgeHit]:
                assert self.currently_open == 1
                return [
                    KnowledgeHit(
                        chunk_id="chunk-1",
                        document_id="doc-1",
                        document_title="Policy",
                        content="Relevant evidence.",
                        score=0.05,
                    )
                ]

        class GeneratorAssertingClosedScope:
            def __init__(self, scope: ScopeRecorder) -> None:
                self._scope = scope

            async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
                assert self._scope.currently_open == 0
                return make_answer(chunks_count=len(chunks))

        scope = ScopeRecorder()
        knowledge_search = InProcessKnowledgeSearchAdapter(SearchKnowledge(scope.open_retriever))
        workflow = LangGraphRagWorkflow(
            knowledge_search,
            GeneratorAssertingClosedScope(scope),
            min_relevance_score=0.028,
        )

        answer = await AskQuestion(workflow).execute("Can I return this?")

        assert len(answer.sources) == 1

    async def test_no_relevant_evidence_produces_an_honest_refusal(self) -> None:
        """THE key RAG behavior: better a refusal than a hallucination."""
        retriever = FakeKnowledgeSearch([make_retrieved_chunk(score=0.001)])
        generator = FakeAnswerGenerator(make_answer())

        answer = await self.build_use_case(retriever, generator).execute(
            "What is the airspeed velocity of an unladen swallow?"
        )

        assert answer.text == REFUSAL_MESSAGE
        assert answer.sources == ()
        assert generator.calls == []  # the LLM was never even called

    async def test_empty_question_is_rejected_before_touching_the_graph(self) -> None:
        use_case = self.build_use_case(FakeKnowledgeSearch([]), FakeAnswerGenerator(make_answer()))
        with pytest.raises(EmptyQuestionError):
            await use_case.execute("   ")

    async def test_question_and_top_k_domain_limits_apply_without_http(self) -> None:
        use_case = self.build_use_case(FakeKnowledgeSearch([]), FakeAnswerGenerator(make_answer()))

        with pytest.raises(InvalidQuestionError, match="at most"):
            await use_case.execute("x" * 4_001)
        with pytest.raises(InvalidQuestionError, match="top_k"):
            await use_case.execute("valid question", top_k=21)

    async def test_retrieval_outage_surfaces_as_a_domain_signal(self) -> None:
        """A retriever that cannot reach its backend raises
        RetrievalUnavailableError; the graph must NOT swallow it into a
        refusal (that would lie: no evidence ≠ unreachable evidence) — the
        signal propagates to the HTTP boundary, which answers 503."""
        use_case = self.build_use_case(
            FailingKnowledgeSearch(RetrievalUnavailableError("provider down")),
            FakeAnswerGenerator(make_answer()),
        )

        with pytest.raises(RetrievalUnavailableError):
            await use_case.execute("Can I return a product?")

    async def test_default_top_k_comes_from_the_constructor(self) -> None:
        """KA_RETRIEVAL_TOP_K wiring: a caller who does not pass top_k gets
        the server default, not a hardcoded one."""
        retriever = FakeKnowledgeSearch([make_retrieved_chunk()])
        generator = FakeAnswerGenerator(make_answer())
        workflow = LangGraphRagWorkflow(retriever, generator, min_relevance_score=0.0)
        use_case = AskQuestion(workflow, default_top_k=3)

        await use_case.execute("q?")
        await use_case.execute("q?", top_k=1)

        assert [limit for _, limit in retriever.calls] == [3, 1]
