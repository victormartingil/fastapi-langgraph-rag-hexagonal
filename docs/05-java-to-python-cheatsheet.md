# Already coming from Java?

Start with
[`victormartingil/python-for-java-devs`](https://github.com/victormartingil/python-for-java-devs).
That project explains the language and ecosystem transition: dataclasses,
Protocols, dependency injection, testing, packaging, async code, and the
Python alternatives to common Java/Spring patterns.

This repository does not repeat that material. It is the advanced continuation:

- `python-for-java-devs` uses modular organization as the pragmatic default;
- this project shows when multiple real integration boundaries justify
  formal ports and adapters;
- two bounded contexts demonstrate ownership without introducing
  microservices;
- LangGraph is isolated as infrastructure rather than allowed to become the
  application architecture;
- tests, RAG evaluations, threat modeling, observability, packaging, and
  supply-chain controls show the path from example code to a reusable
  engineering reference.

Continue here:

1. [Architecture overview](00-architecture-overview.md)
2. [Bounded-context ownership](08-bounded-context-ownership.md)
3. [Pythonic ports and adapters](09-pythonic-ports-and-adapters.md)
4. [LangGraph orchestration adapter](03-langgraph-orchestration.md)

The main translation principle is simple: preserve the engineering purpose of
SOLID and clean boundaries, but express it with modules, functions,
structural typing, dataclasses, context managers, and explicit composition
instead of recreating Spring's ceremony.
