"""Unit tests for the RAG graph: nodes, routing, and the full compiled flow.

The graph is tested against fake ports — LangGraph itself needs no Docker and
no LLM — including the most important behavior of the whole system: when no
relevant context survives grading, the answer is an honest refusal.
"""

import pytest

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
    RetrievalUnavailableError,
)
from knowledge_assistant.assistant.domain.models import Answer, RetrievedChunk, Source
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
