# 01 — Hexagonal Architecture in Python

> How this project applies ports & adapters the *Pythonic* way — no
> interfaces-in-everything, no framework magic, and rules that are enforced
> by tests instead of by convention.

## The one rule

Dependencies may only point **inward**:

```
        ┌─────────────────────────────────────────────┐
        │                  adapters                   │
        │  FastAPI routers, SQLAlchemy, pgvector SQL, │
        │  Ollama/OpenAI clients, Pydantic AI         │
        │        │  (depends on ↓)                    │
        │  ┌───────────────────────────────────────┐  │
        │  │            application                │  │
        │  │  use cases + ports (Protocols) +      │  │
        │  │  the LangGraph RAG pipeline           │  │
        │  │        │  (depends on ↓)              │  │
        │  │  ┌───────────────────────────────┐    │  │
        │  │  │            domain             │    │  │
        │  │  │  frozen dataclasses, value    │    │  │
        │  │  │  objects, pure services       │    │  │
        │  │  └───────────────────────────────┘    │  │
        │  └───────────────────────────────────────┘  │
        └─────────────────────────────────────────────┘
```

The domain imports **nothing** outside the standard library. The application
layer imports the domain. Adapters import both. Never the reverse.

This is not enforced by discipline — it is enforced by
[import-linter](https://import-linter.readthedocs.io/) contracts in
`pyproject.toml`, executed in `tests/architecture/`. Try importing SQLAlchemy
from a domain module: the build goes red.

## The four building blocks, as used here

### 1. Domain models — frozen dataclasses

```python
@dataclass(frozen=True, slots=True)
class Chunk:
    id: DocumentId
    text: ChunkText
    position: int
    embedding: EmbeddingVector | None = None
```

Why frozen dataclasses and not Pydantic? Because the domain should not depend
on any framework — not even a validation framework. Immutability (`frozen`)
gives us value semantics for free, `slots=True` avoids per-instance attribute
dictionaries, and `__post_init__` gives us just enough validation to make
invalid states unrepresentable (`ChunkText("")` raises).

### 2. Ports — `typing.Protocol`

```python
class DocumentRepository(Protocol):
    async def save(self, document: Document) -> None: ...
    async def get_by_id(self, document_id: DocumentId) -> Document | None: ...
```

A Protocol is a **structural** interface: any class with matching methods
satisfies it, without inheriting anything. This is the Pythonic answer to
Java/C# interfaces — defined by the *consumer* (the use case), implemented by
the *provider* (the adapter), checked by mypy.

Ports live in `*/application/ports.py`, one small Protocol per volatile
boundary (Interface Segregation). Embeddings belong to `knowledge_base`;
the assistant sees only its own `KnowledgeSearch` port.

### 3. Use cases — small verb-named classes

```python
class IngestDocument:
    def __init__(self, open_repository: OpenRepository,
                 embedding_provider: EmbeddingProvider,
                 text_extractors: Sequence[TextExtractor], ...): ...

    async def execute(self, file_name: str, data: bytes, ...) -> Document:
        extractor = self._pick_extractor(file_name)
        raw_text = extractor.extract(file_name, data)
        chunk_texts = chunk_text(raw_text, ...)
        embeddings = await self._embedding_provider.embed(...)
        ...
```

(`OpenRepository` is itself a port — a factory of short-lived repository
scopes — so the use case never holds a database connection across the slow
embedding calls; see [ADR-0005](adr/0005-short-transaction-ingest.md).)

The use case reads like the feature description because it only orchestrates
domain logic and ports. It cannot contain SQL or HTTP — it has nothing to
write them with.

### 4. Adapters — technology-prefixed classes

```
SqlAlchemyDocumentRepository   PgVectorHybridRetriever
OllamaEmbeddingProvider        OpenAiEmbeddingProvider
PdfTextExtractor               PlainTextExtractor
PydanticAiAnswerGenerator
```

The prefix makes the coupling visible in the name. Adapters implement ports
**structurally** (no `class X(DocumentRepository)`), and all vendor knowledge
is quarantined inside them. Swapping Ollama for OpenAI is a `.env` change;
swapping PostgreSQL for Qdrant is one new file plus one line in
`bootstrap.py`.

## Explicit mappers at every boundary

Nothing crosses a layer boundary without a named, reviewable function:

| Boundary                         | Mapper                                          |
| -------------------------------- | ----------------------------------------------- |
| domain ↔ SQLAlchemy ORM          | `persistence/mappers.py`                        |
| domain → HTTP response schemas   | `http/mappers.py`                               |
| domain ↔ Pydantic AI payloads    | inside `llm/pydantic_ai.py`                     |

Yes, it is more code than returning the ORM object directly. That is the
point: the mapping is where schema changes stop propagating.

## The composition root

`src/knowledge_assistant/bootstrap.py` is the **only** module that knows
which concrete classes exist. It does two things:

1. `build_container(settings)` creates long-lived adapters (engine, HTTP
   clients, providers).
2. `provide_*` functions are FastAPI dependencies that assemble per-request
   use cases from a fresh session + the long-lived adapters.

Routers never construct anything; they declare
`use_case: Annotated[IngestDocument, Depends(container.provide_ingest_document)]`.

This gives you constructor injection without a DI framework, and it makes
testing trivial: tests either call use cases with fakes directly (unit) or
override the `provide_*` dependency (e2e).

## Bounded contexts

`knowledge_base` owns document lifecycle and search; `assistant` owns grounded
Q&A. The assistant does not know its tables or persistence models. Its
`KnowledgeSearch` port is implemented by one in-process adapter that calls the
public `SearchKnowledge` use case. Import-linter allows only that explicit
bridge. `shared_kernel` remains deliberately tiny: shared value objects and
domain-level error types, not services or vendor abstractions.

## Further reading

- *Architecture Patterns with Python* (Percival & Gregory) — the style guide
  for this codebase.
- `docs/adr/0002-langgraph-as-orchestration-adapter.md` — where the orchestrator
  sits and why.
- `docs/05-java-to-python-cheatsheet.md` — if you come from Spring/Java.
