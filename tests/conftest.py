"""Root test fixtures: a real PostgreSQL+pgvector, on demand, via testcontainers.

Only integration and e2e tests request `postgres_database_url`, so the unit
and architecture suites run with ZERO Docker dependency — a property we
verify in CI by running them on a plain runner.

The fixture is session-scoped: one container (and one Alembic migration run)
serves all integration and e2e tests in the session.
"""

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_migrations(database_url: str, extra_env: dict[str, str] | None = None) -> None:
    """Apply Alembic migrations against the throwaway database.

    This is itself a test of the migrations: if the schema cannot be built
    from scratch, every integration/e2e test fails loudly here. `extra_env`
    forwards migration-time configuration (e.g. KA_FTS_LANGUAGE for the
    multilingual test container).
    """
    alembic = shutil.which("alembic") or str(Path(sys.executable).parent / "alembic")
    env = {**os.environ, "KA_DATABASE_URL": database_url, **(extra_env or {})}
    try:
        subprocess.run(
            [alembic, "upgrade", "head"],
            cwd=PROJECT_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Alembic migrations failed against test database:\n{exc.stdout}\n{exc.stderr}"
        ) from exc


@pytest.fixture(scope="session")
def postgres_database_url() -> Iterator[str]:
    """Start pgvector/pgvector:0.8.1-pg16, migrate it, yield its async SQLAlchemy URL.

    The image is pinned to the same tag as docker-compose.yml (verified on
    Docker Hub) so tests and the local stack run the same database.
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer("pgvector/pgvector:0.8.1-pg16") as postgres:
        host = postgres.get_container_host_ip()
        port = postgres.get_exposed_port(5432)
        url = (
            f"postgresql+asyncpg://{postgres.username}:{postgres.password}"
            f"@{host}:{port}/{postgres.dbname}"
        )
        _run_migrations(url)
        yield url
