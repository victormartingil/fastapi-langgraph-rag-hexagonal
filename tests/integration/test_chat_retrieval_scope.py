"""Integration tests for chat retrieval transaction boundaries."""

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from knowledge_assistant import bootstrap
from knowledge_assistant.assistant.adapters.outbound.knowledge.in_process import (
    InProcessKnowledgeSearchAdapter,
)
from knowledge_assistant.assistant.adapters.outbound.orchestration.langgraph.builder import (
    LangGraphRagWorkflow,
)
from knowledge_assistant.assistant.application.ask import AskQuestion
from knowledge_assistant.assistant.domain.models import Answer, RetrievedChunk, Source
from knowledge_assistant.knowledge_base.application.ingest import IngestDocument
from knowledge_assistant.knowledge_base.application.queries import SearchKnowledge
from knowledge_assistant.knowledge_base.domain.value_objects import EmbeddingVector
from knowledge_assistant.platform.database.session import session_scope

pytestmark = pytest.mark.integration


class ConstantEmbeddingProvider:
    async def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [EmbeddingVector(tuple([0.5] * 768)) for _ in texts]


class PlainTextOnlyExtractor:
    def supports(self, file_name: str) -> bool:
        return file_name.endswith(".txt")

    def extract(self, file_name: str, data: bytes) -> str:
        return data.decode()


class BlockingAnswerGenerator:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate(self, question: str, chunks: list[RetrievedChunk]) -> Answer:
        self.started.set()
        await self.release.wait()
        return Answer(
            text="Grounded answer.",
            sources=tuple(
                Source(
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    chunk_id=chunk.chunk_id,
                    excerpt=chunk.content,
                    score=chunk.score,
                )
                for chunk in chunks
            ),
        )


async def test_chat_releases_the_database_connection_before_generation(
    postgres_database_url: str,
) -> None:
    engine = create_async_engine(
        postgres_database_url,
        pool_pre_ping=True,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.2,
    )
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    embedding_provider = ConstantEmbeddingProvider()

    async with session_factory() as cleanup:
        await cleanup.execute(text("DELETE FROM chunks"))
        await cleanup.execute(text("DELETE FROM documents"))
        await cleanup.commit()

    ingest = IngestDocument(
        open_repository=bootstrap.repository_scope_factory(session_factory),
        embedding_provider=embedding_provider,
        text_extractors=(PlainTextOnlyExtractor(),),
        expected_embedding_dimension=768,
        embedding_model_name="test-embedding",
    )
    await ingest.execute(
        title="Return Policy",
        file_name="policy.txt",
        data=b"Products can be returned within 30 days with the receipt.",
    )

    open_retriever = bootstrap.retriever_scope_factory(
        session_factory,
        embedding_provider,
        fetch_limit=20,
        rrf_k=60,
        tsconfig="english",
    )
    knowledge_search = InProcessKnowledgeSearchAdapter(SearchKnowledge(open_retriever))
    generator = BlockingAnswerGenerator()
    workflow = LangGraphRagWorkflow(
        knowledge_search,
        generator,
        min_relevance_score=0.0,
    )
    ask_question = AskQuestion(workflow)

    chat_task = asyncio.create_task(ask_question.execute("Can I return it?"))
    try:
        await asyncio.wait_for(generator.started.wait(), timeout=2)

        async with session_scope(session_factory) as session:
            result = await asyncio.wait_for(session.execute(text("SELECT 1")), timeout=1)
            assert result.scalar_one() == 1
    finally:
        generator.release.set()
        answer = await chat_task
        await engine.dispose()

    assert len(answer.sources) == 1
