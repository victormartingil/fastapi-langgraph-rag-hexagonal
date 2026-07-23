"""Integration tests: the FTS-language parity guard against a real database.

Migration 0004 records the language the `tsv` column was ACTUALLY built with
(introspected from the column expression, not copied from the environment).
The startup guard must:

- pass when configuration and schema agree,
- fail fast when the recorded language was tampered with (drift, or a
  database migrated with a different KA_FTS_LANGUAGE),
- fail with "run the migrations" when the row is absent.

The tamper test restores the original value: integration fixtures share one
container per session, so leaking the tamper would poison other tests.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from knowledge_assistant.platform.database.schema_meta import (
    assert_fts_language_parity,
)

pytestmark = pytest.mark.integration


class TestFtsLanguageParityGuard:
    async def test_parity_passes_on_a_correctly_migrated_database(
        self, session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        # The session container is migrated with the default: english.
        await assert_fts_language_parity(session_factory, expected="english")

    async def test_tampered_meta_fails_fast(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        # Committed via a SEPARATE session: the guard opens its own
        # transaction and would never see uncommitted tampering.
        async with session_factory() as tamper:
            await tamper.execute(
                text("UPDATE schema_meta SET value = 'german' WHERE key = 'fts_language'")
            )
            await tamper.commit()
        try:
            with pytest.raises(ValueError, match="'german'"):
                await assert_fts_language_parity(session_factory, expected="english")
        finally:
            async with session_factory() as restore:
                await restore.execute(
                    text("UPDATE schema_meta SET value = 'english' WHERE key = 'fts_language'")
                )
                await restore.commit()

    async def test_missing_row_means_run_the_migrations(
        self,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        async with session_factory() as tamper:
            await tamper.execute(text("DELETE FROM schema_meta WHERE key = 'fts_language'"))
            await tamper.commit()
        try:
            with pytest.raises(RuntimeError, match="alembic upgrade head"):
                await assert_fts_language_parity(session_factory, expected="english")
        finally:
            async with session_factory() as restore:
                await restore.execute(
                    text("INSERT INTO schema_meta (key, value) VALUES ('fts_language', 'english')")
                )
                await restore.commit()
