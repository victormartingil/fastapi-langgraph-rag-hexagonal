# Pythonic ports and adapters

This project uses hexagonal architecture as a dependency rule, not as a
reason to reproduce Spring in Python.

## When hexagonal architecture is worth its cost

Use it when the application has important policy that must survive changes to
multiple external mechanisms. This repository has real variation in:

- Ollama and OpenAI embedding providers;
- text and PDF extraction, with more formats expected;
- SQL persistence and hybrid retrieval;
- in-process or future remote knowledge access;
- Pydantic AI generation;
- LangGraph or another workflow runtime;
- FastAPI as an inbound delivery mechanism.

For a CRUD service with one database and no meaningful policy, a domain-based
module with direct framework use is usually the better design. AWS makes the
same trade-off explicit in its
[hexagonal architecture guidance](https://docs.aws.amazon.com/prescriptive-guidance/latest/cloud-design-patterns/hexagonal-architecture.html):
testability and replaceability are benefits, but extra interfaces are a cost.

## Protocols at volatile boundaries only

Ports use [`typing.Protocol`](https://peps.python.org/pep-0544/) and structural
typing:

```python
class KnowledgeSearch(Protocol):
    async def search(self, question: str, limit: int) -> list[RetrievedChunk]: ...
```

An adapter satisfies the port by having the right behavior and signature. It
does not inherit from the Protocol and does not write `implements`.

This gives the Dependency Inversion Principle without nominal-interface
ceremony. It also keeps the interface where its consumer owns it:
`KnowledgeSearch` belongs to the assistant, not to PostgreSQL or the
knowledge base.

There is intentionally no Protocol for:

- pure policies such as `filter_relevant_evidence`;
- domain dataclasses;
- the `Settings` object;
- trivial mappers;
- classes with only one stable internal caller.

An abstraction must isolate a plausible source of change or enable a useful
test seam.

## Modules follow cohesion, not “one class per file”

The application is organized by use case:

- `knowledge_base/application/ingest.py` owns the complex ingestion slice;
- `knowledge_base/application/queries.py` groups small read use cases;
- `assistant/application/ask.py` owns question validation and workflow entry;
- `assistant/application/policies.py` groups pure grading, routing, and
  refusal decisions.

Python modules are the first encapsulation tool. Empty factories, getters,
setters, manager classes, and one-file-per-class layouts would increase
navigation cost without protecting an invariant.

## Functional core, imperative shell

Pure code makes decisions:

- value objects reject invalid state;
- chunking transforms text deterministically;
- grading filters evidence;
- routing chooses generate or refuse;
- citation invariants live in domain values and adapter validation.

The shell coordinates effects:

- FastAPI parses and renders HTTP;
- application use cases call ports;
- adapters perform SQL, extraction, HTTP, and model calls;
- LangGraph schedules the assistant workflow;
- the lifespan owns startup and cleanup.

Not every use case is a pure function. `IngestDocument` is an imperative
orchestrator, but its decisions and boundaries are explicit and independently
testable.

## Dataclasses inside, Pydantic at boundaries

Domain values use immutable slotted dataclasses:

```python
@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: tuple[Source, ...]
    is_refusal: bool = False
```

Pydantic is used where parsing untrusted structures is its actual job:

- HTTP request/response schemas;
- environment configuration;
- structured LLM output inside the Pydantic AI adapter.

ORM models remain inside persistence. Named mappers translate at every
boundary. No Pydantic or SQLAlchemy object crosses into a domain model.

## Composition without a DI framework

`bootstrap.py` is the composition root:

1. `build_container(settings)` chooses and creates process-lived concrete
   adapters;
2. small FastAPI providers retrieve that container from `app.state`;
3. providers assemble request-scoped use cases and database sessions.

Routers depend on use-case types through `Annotated[..., Depends(...)]`.
They never instantiate a repository, provider, workflow, or HTTP client.

Tests either construct a use case directly with manual fakes or override the
same provider seam FastAPI uses. A third-party DI container would add another
lifecycle and debugging model without solving a missing problem.

## Concurrency and resource ownership

- one SQLAlchemy `AsyncSession` is used by one task only;
- ingestion opens short transaction scopes around DB work, never around
  extraction or embeddings;
- blocking extraction runs in a worker thread;
- HTTP clients and the engine are process-owned and closed during lifespan
  shutdown;
- shutdown keeps closing later resources even when an earlier close fails or
  the shutdown task is cancelled;
- retries are bounded and occur only at provider boundaries;
- cancellation is not converted into success or a fallback answer;
- no task is launched fire-and-forget inside the web process.

`TaskGroup` is reserved for genuinely independent work. Sequential ingestion
batches are intentional because they bound provider pressure and preserve
simple failure semantics.

## Exception design

Exceptions are grouped by how callers handle them, consistent with
[PEP 8](https://peps.python.org/pep-0008/#exceptions):

- invalid domain input → typed 4xx;
- transient provider/database exhaustion → typed 503;
- invalid grounded model output after retries → typed 502;
- permanent configuration or programming failures remain loud.

This is why an outage exception may be translated at a context bridge: the
same technical failure has different meaning to the knowledge-base and
assistant APIs.

## Deliberately absent abstractions

There is no generic repository, service base class, command bus, unit-of-work
framework, event bus, broker, provider registry, or abstract factory.

Those patterns are not forbidden. They are deferred until a second concrete
use case creates evidence that the abstraction reduces change rather than
redistributes it.
