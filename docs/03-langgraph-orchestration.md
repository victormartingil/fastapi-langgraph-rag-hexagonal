# 03 — LangGraph Orchestration

> Why the read side is a graph, how it is wired, and why LangGraph lives in
> the application layer.

## Why a graph at all?

The Q&A flow is not a straight line — it has a **decision**:

```
START → retrieve → grade ──┬─ evidence found ─► generate → END
                           └─ no evidence ────► refuse   → END
```

You could write that as an `if` in a service method. For this three-node
pipeline that would honestly be fine. The reasons to reach for LangGraph are
the ones this project is designed to *teach and grow into*:

- the flow is explicit and inspectable (each node is a small pure function,
  each edge is visible);
- adding nodes (a query rewriter, an LLM grader, a router for ad-hoc
  documents — see the roadmap) is additive, not a rewrite;
- conditional routing, streaming, checkpointing and multi-turn memory are
  framework features you opt into later, not patterns you hand-roll.

## The pieces

### State — `graph/state.py`

A `TypedDict` passed through the graph; nodes return **partial updates**:

```python
class RagState(TypedDict, total=False):
    question: str
    top_k: int
    retrieved_chunks: list[RetrievedChunk]
    relevant_chunks: list[RetrievedChunk]
    answer: Answer
```

It carries plain domain objects — no LangGraph types leak into the domain.

### Nodes — `graph/nodes.py`

Nodes are built by **factory functions that close over ports**:

```python
def make_retrieve_node(retriever: ChunkRetriever) -> Node:
    async def retrieve(state: RagState) -> dict[str, object]:
        chunks = await retriever.retrieve(state["question"], limit=state["top_k"])
        return {"retrieved_chunks": chunks}
    return retrieve
```

This is dependency injection without LangGraph knowing anything about DI:
the closure captures the port, the node signature stays `(state) -> update`.
Every node is unit-tested with a hand-written fake — no Docker, no LLM.

The four nodes:

| Node      | Calls                          | Purpose                              |
| --------- | ------------------------------ | ------------------------------------ |
| `retrieve`  | `ChunkRetriever` port          | hybrid search in the knowledge base  |
| `grade`     | nothing (pure function)        | drop chunks below the relevance bar  |
| `generate`  | `AnswerGenerator` port         | grounded, cited answer via the LLM   |
| `refuse`    | nothing                        | fixed honest refusal, zero LLM cost  |

### Routing — `route_after_grading`

```python
def route_after_grading(state: RagState) -> str:
    return "generate" if state.get("relevant_chunks") else "refuse"
```

A conditional edge: the graph's control flow is data-driven. This single
function encodes the system's core value — *no evidence, no answer*.

### Assembly — `graph/builder.py`

`build_rag_graph(retriever, answer_generator, min_relevance_score=...)` wires
nodes and edges and returns the **uncompiled** graph; the caller compiles it
(`.compile()`). That keeps the wiring testable and leaves the door open to
attaching a checkpointer later without touching this code (roadmap item 4).

### The use case — `application/ask.py`

`AskQuestion` is the entry point the HTTP layer depends on. It validates the
question, invokes the compiled graph, and unwraps the final `Answer`. It
knows the *shape* of the pipeline; it does not know PostgreSQL or Ollama
exist.

## Why the graph is in the application layer

"Retrieve, grade, then answer or refuse" is **application policy** — it would
exist with any orchestrator, any database, any LLM. LangGraph is used here as
a *library inside the application layer*, exactly like SQLAlchemy is used as
a library inside infrastructure adapters. The domain never imports it
(enforced by import-linter), and the only infrastructure that knows about the
graph's existence is the composition root, which injects the ports.

See `docs/adr/0002-langgraph-in-application-layer.md` for the decision record.

## Extending the graph (guided exercise)

Add an LLM-based grader between `retrieve` and `grade`:

1. Define an `AnswerGrader` Protocol in `application/ports.py`
   (`async def grade(question, chunks) -> list[RetrievedChunk]`).
2. Implement `PydanticAiAnswerGrader` in `infrastructure/llm/`.
3. Add a `make_llm_grade_node(grader)` factory and insert the node between
   `retrieve` and `grade` in `builder.py`.
4. Write unit tests with a fake grader; run the architecture tests to prove
   the layers survived.
