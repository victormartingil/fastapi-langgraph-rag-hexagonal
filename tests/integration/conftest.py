"""Integration test fixtures: real PostgreSQL+pgvector via testcontainers.

The `postgres_database_url` session fixture (root conftest) starts ONE
container and runs the Alembic migrations. Here we give each test a clean
session and a clean database.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_assistant.platform.database.session import (
    create_engine,
    create_session_factory,
    session_scope,
)

pytestmark = pytest.mark.integration


@pytest.fixture
async def session_factory(
    postgres_database_url: str,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """The session factory itself, for tests that need SEVERAL independent
    transactions (e.g. the dedup race: one transaction commits, another
    loses)."""
    engine = create_engine(postgres_database_url)
    session_factory = create_session_factory(engine)

    async with session_factory() as cleanup:
        await cleanup.execute(text("DELETE FROM chunks"))
        await cleanup.execute(text("DELETE FROM documents"))
        await cleanup.commit()

    yield session_factory

    await engine.dispose()


@pytest.fixture
async def session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """A committed-on-success session against a freshly emptied database."""
    async with session_scope(session_factory) as session:
        yield session
