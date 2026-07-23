"""Startup parity guard for schema-bound configuration (ADR-0004).

Some configuration is baked INTO the schema at migration time: the FTS
language lives in the generated `tsv` column's expression. If the app runs
with a different language than the schema was built with, full-text search
silently degrades — the worst kind of failure. Migration 0004 therefore
records the truth in the `schema_meta` table (introspected from the actual
column, not copied from the environment), and this module verifies it at
startup:

- mismatch            -> refuse to boot (ValueError naming both languages
                         and the fix);
- table missing       -> refuse to boot (schema older than the app: run
                         `alembic upgrade head`);
- database unreachable -> log a warning and boot anyway: availability is
                         already reported honestly by /health, and a transient
                         DB outage must not become a boot loop.
"""

import structlog
from sqlalchemy import text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = structlog.get_logger()

_FTS_LANGUAGE_SQL = text("SELECT value FROM schema_meta WHERE key = 'fts_language'")

# PostgreSQL error code for undefined_table — the ONLY ProgrammingError the
# guard may reinterpret. asyncpg exposes the code as `sqlstate` on the
# original exception inside SQLAlchemy's wrapper.
_UNDEFINED_TABLE_PGCODE = "42P01"


def _is_missing_meta_table(exc: ProgrammingError) -> bool:
    """True iff the error is specifically 'relation schema_meta does not
    exist'. Other ProgrammingErrors — insufficient_privilege (42501), syntax
    errors — are operational or bug signals and must surface honestly, not
    be misreported as 'run the migrations'."""
    return getattr(exc.orig, "sqlstate", None) == _UNDEFINED_TABLE_PGCODE


def _assert_language_matches(recorded: str, expected: str) -> None:
    """Pure comparison, unit-tested directly: the error must name BOTH
    languages and the way out — an operator reading it should not have to
    guess."""
    if recorded != expected:
        msg = (
            f"FTS language mismatch: the database schema was built for "
            f"{recorded!r} (schema_meta.fts_language) but KA_FTS_LANGUAGE="
            f"{expected!r}. Set KA_FTS_LANGUAGE={recorded!r} to match the "
            f"database, or rebuild the schema on a fresh database with "
            f"KA_FTS_LANGUAGE={expected!r} (`uv run alembic upgrade head`). "
            "See docs/adr/0004."
        )
        raise ValueError(msg)


async def assert_fts_language_parity(
    session_factory: async_sessionmaker[AsyncSession], *, expected: str
) -> None:
    """Verify the migrated schema speaks the configured FTS language."""
    try:
        async with session_factory() as session:
            recorded = await session.scalar(_FTS_LANGUAGE_SQL)
    except ProgrammingError as exc:
        if not _is_missing_meta_table(exc):
            raise  # permission errors, syntax errors: surface them honestly
        # schema_meta does not exist: the database was migrated with an older
        # version of the app. That is a mismatch-class failure, not an outage.
        msg = (
            "The schema_meta table is missing: the database schema is older "
            "than this application. Run `uv run alembic upgrade head` first."
        )
        raise RuntimeError(msg) from exc
    except OperationalError:
        logger.warning(
            "fts_language_parity_check_skipped",
            reason="database unreachable at startup; /health reports availability",
        )
        return

    if recorded is None:
        msg = (
            "schema_meta.fts_language is unset: the database was not migrated "
            "by migration 0004 or later. Run `uv run alembic upgrade head`."
        )
        raise RuntimeError(msg)
    _assert_language_matches(recorded, expected)
