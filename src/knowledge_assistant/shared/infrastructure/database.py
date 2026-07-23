"""Database plumbing: async engine + session factory.

One engine per process (it manages the connection pool), one session per
request. The composition root creates both; FastAPI dependencies hand out
sessions and guarantee cleanup.

pgvector note: SQLAlchemy's `Vector` column type serializes vectors to their
canonical text form ('[0.1, 0.2, ...]') on write and parses them back on
read, and PostgreSQL casts text to `vector` on assignment — so NO asyncpg
codec registration is needed. (Registering `pgvector.asyncpg.register_vector`
would actually break this: the codec expects raw lists while the SQLAlchemy
type already produced strings.)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def is_db_outage_error(exc: BaseException) -> bool:
    """True iff the failure means the DATABASE is unreachable — a 503-class
    outage, never a 500-class bug.

    Three shapes, one meaning:

    - `OperationalError` / `InterfaceError`: SQLAlchemy-wrapped DBAPI
      failures (auth rejected, connection dropped mid-query, ...);
    - raw `OSError`: asyncpg's CONNECT path does NOT raise DBAPI errors —
      "connection refused/unreachable" escapes SQLAlchemy's translation as
      a bare OSError (asyncio's happy-eyeballs "Multiple exceptions"
      wrapper included). Verified against asyncpg 0.31 + SQLAlchemy 2.0.

    Note `TimeoutError` IS an OSError: a pool-acquisition or connect
    timeout is exactly the "database overwhelmed/unreachable" signal this
    predicate exists for. SQL BUGS (ProgrammingError and friends) are not
    OSError and correctly stay out.
    """
    return isinstance(exc, OperationalError | InterfaceError | OSError)


def create_engine(database_url: str) -> AsyncEngine:
    """Create the async engine with liveness checks on pooled connections."""
    return create_async_engine(database_url, pool_pre_ping=True)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Sessions expire on commit=False so domain mapping after commit is safe."""
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def session_scope(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Transaction boundary for one unit of work: commit on success, rollback
    on any exception, always close. Use cases get a session from here."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
