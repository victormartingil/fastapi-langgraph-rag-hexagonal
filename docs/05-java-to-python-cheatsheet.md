# 05 — Java to Python Cheatsheet

> Coming from Spring/Java? This project was built with you in mind. Every
> Java reflex has an idiomatic Python counterpart — usually less code, always
> explicit.

| Java / Spring                        | Python (this repo)                                  | Notes |
| ------------------------------------ | --------------------------------------------------- | ----- |
| `record`                             | `@dataclass(frozen=True)`                            | Value semantics; validation in `__post_init__` |
| `interface`                          | `typing.Protocol`                                    | **Structural**: no `implements`, no inheritance |
| `class X implements Y`               | just write matching methods                          | mypy checks conformance statically |
| ArchUnit                             | **import-linter** contracts (`pyproject.toml`)       | Executed in `tests/architecture/` |
| Spring `@Configuration` / DI container | `bootstrap.py` (composition root) + `Depends`      | Constructor injection, no framework magic |
| `@Autowired`                         | constructor parameter with a Protocol type           | Wired in one place: `bootstrap.py` |
| Flyway / Liquibase                   | **Alembic** (`platform/database/migrations/`)    | `alembic upgrade head` |
| JPA `@Entity`                        | SQLAlchemy `Mapped[...]` models (infrastructure!)    | Never returned to upper layers |
| MapStruct / manual converters        | `mappers.py` modules                                 | Plain functions, unit-tested |
| `@RestController`                    | FastAPI router (thin HTTP adapter)                   | Schemas in Pydantic, domain in dataclasses |
| DTO                                  | Pydantic `BaseModel` schema in `http/schemas.py`     | Validation at the boundary |
| `@Transactional`                     | `session_scope()` context manager                    | Commit on success, rollback on exception |
| Maven / Gradle                       | **uv** (`pyproject.toml` + `uv.lock`)                | `uv sync`, `uv run ...` |
| JUnit 5                              | **pytest**                                           | Functions, fixtures, `parametrize` |
| Mockito                              | hand-written fakes (`tests/unit/fakes.py`)           | Working implementations, not call assertions |
| Testcontainers Java                  | `testcontainers[postgres]`                           | Same idea, same Docker requirement |
| SLF4J + MDC                          | **structlog** + correlation-ID middleware            | Context bound per request via contextvars |
| Resilience4j `@Retry` / circuit breaker | **tenacity** + graceful fallback in the LLM adapter | Retries with exponential backoff |
| Strategy pattern (interface + beans) | Protocol + one adapter per vendor, chosen in `bootstrap.py` | e.g. Ollama vs OpenAI |
| Chain of Responsibility              | `Sequence[TextExtractor]` with `supports()`          | First capable extractor wins |
| Lombok                               | `@dataclass`                                         | (but write it out — didactic repo) |
| `application.yml` + `@ConfigurationProperties` | `pydantic-settings` `BaseSettings`     | Typed, validated, `.env`-backed |

## The three reflexes to unlearn

1. **Stop writing interfaces for everything.** A Protocol only where a seam
   is real (ports). Concrete classes elsewhere.
2. **Stop reaching for a framework.** DI is a module (`bootstrap.py`).
   Transactions are a context manager. Middleware is a class.
3. **Stop hiding mapping behind annotations.** The mappers are plain
   functions you can breakpoint, review, and unit-test.

## Where to look first

| You'd look for…               | Go to                                                              |
| ----------------------------- | ------------------------------------------------------------------ |
| the "service layer"           | `knowledge_base/application/ingest.py`, `queries.py`, `assistant/application/ask.py` |
| the "repository interfaces"   | `*/application/ports.py`                                           |
| the "JPA entities"            | `knowledge_base/adapters/outbound/persistence/models.py`                   |
| the "Spring config"           | `bootstrap.py`, `config.py`                                        |
| the "integration tests"       | `tests/integration/`, `tests/e2e/`                                 |
