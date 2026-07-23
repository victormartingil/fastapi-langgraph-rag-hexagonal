"""Alembic environment (async, asyncpg).

The URL resolution order is deliberate:

1. `KA_DATABASE_URL` environment variable — used by docker-compose and by the
   integration/e2e tests (which point Alembic at a testcontainers database).
2. The `sqlalchemy.url` from alembic.ini — a local-development default.

`.env` is loaded FIRST (without overriding real environment variables — the
same precedence pydantic-settings gives the app) so migrate-time and run-time
resolve the SAME configuration from the SAME source: KA_FTS_LANGUAGE in `.env`
now reaches migration 0003 exactly as it reaches Settings. This removes the
drift window where the schema is built for one language and the app queries
with another (the startup parity guard in schema_meta.py remains as
defense-in-depth for cross-environment drift).

Migrations are pure infrastructure, so this file may import SQLAlchemy freely.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from knowledge_assistant.knowledge_base.adapters.outbound.persistence.models import Base

load_dotenv()  # override=False: real environment variables keep precedence

config = context.config

if url := os.environ.get("KA_DATABASE_URL"):
    config.set_main_option("sqlalchemy.url", url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate diffs are computed against this metadata.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL without a database connection (alembic upgrade --sql)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    def do_run_migrations(sync_connection: Connection) -> None:
        # Alembic's API is synchronous; SQLAlchemy hands us the sync facade
        # of the async connection via `run_sync`.
        context.configure(connection=sync_connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
