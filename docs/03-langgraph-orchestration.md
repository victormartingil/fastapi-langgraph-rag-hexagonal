# 03 — LangGraph Orchestration

> Why the assistant uses a graph, how it is wired, and how LangGraph remains
> a replaceable orchestration adapter.

## Why a graph at all?

The Q&A flow is not a straight line — it has a **decision**:

```
START → retrieve → grade ──┬─ evidence found ─► generate → END
                           └─ no evidence ────► refuse   → END
```

You could write that as an `if` in a service method. For this three-node
pipeline that would honestly be fine. The reasons to reach for LangGraph are
the ones this project is designed to *teach and grow into*:

- the flow is explicit and inspectable (each node is a small async unit, each
  edge is visible, and decisions remain pure policies);
- adding nodes (a query rewriter, an LLM grader, a router for ad-hoc
  documents — see the roadmap) is additive, not a rewrite;
- conditional routing, streaming, checkpointing and multi-turn memory are
  framework features you opt into later, not patterns you hand-roll.

## The pieces

### State — `adapters/outbound/orchestration/langgraph/state.py`

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

### Nodes — `adapters/outbound/orchestration/langgraph/nodes.py`

Nodes are built by **factory functions that close over ports**:

```python
def make_retrieve_node(search: KnowledgeSearch) -> Node:
    async def retrieve(state: RagState) -> dict[str, object]:
        chunks = await search.search(state["question"], limit=state["top_k"])
        return {"retrieved_chunks": chunks}
    return retrieve
```

This is dependency injection without LangGraph knowing anything about DI:
the closure captures the port, the node signature stays `(state) -> update`.
Every node is unit-tested with a hand-written fake — no Docker, no LLM.

The four nodes:

| Node      | Calls                          | Purpose                              |
| --------- | ------------------------------ | ------------------------------------ |
| `retrieve`  | `KnowledgeSearch` port          | hybrid search in the knowledge base  |
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

### Assembly — `adapters/outbound/orchestration/langgraph/builder.py`

`build_rag_graph(retriever, answer_generator, min_relevance_score=...)` wires
nodes and edges and returns the **uncompiled** graph; the caller compiles it
(`.compile()`). That keeps the wiring testable and leaves the door open to
attaching a checkpointer later without changing application or domain code.

### The use case — `application/ask.py`

`AskQuestion` is the entry point the HTTP layer depends on. It validates the
question and calls the application-owned `RagWorkflow` port. It does not know
LangGraph, PostgreSQL, or an LLM vendor exists.

## Why LangGraph is an adapter

"Retrieve, grade, then answer or refuse" is application policy. The pure
filtering, routing, and refusal rules therefore live in
`assistant/application/policies.py`. LangGraph state, nodes, edges, and
compilation are runtime mechanics under the outbound adapter. Import-linter
forbids LangGraph from domain and application.

See `docs/adr/0002-langgraph-as-orchestration-adapter.md` for the decision record.

## Extend it yourself

The [advanced exercises](11-advanced-exercises.md) include checkpointer,
remote knowledge-adapter, and semantic-grader extensions with acceptance
questions but no implementation recipe.
