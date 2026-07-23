"""Architecture tests, part 2: conventions import-linter cannot express.

1. Infrastructure ADAPTER classes carry a technology prefix
   (`SqlAlchemy*`, `PgVector*`, `Ollama*`, `OpenAi*`, `Pdf*`, `PlainText*`,
   `PydanticAi*`) so their vendor coupling is visible in the name.
2. Port modules contain ONLY Protocol classes.
3. HTTP schemas/routers and ORM models are exempt from rule 1 (they are
   already framework-shaped by definition).
"""

import importlib
import inspect
import pkgutil
from collections.abc import Iterator

import pydantic

ADAPTER_PREFIXES = (
    "SqlAlchemy",
    "PgVector",
    "Ollama",
    "OpenAi",
    "Pdf",
    "PlainText",
    "PydanticAi",
    "InProcess",
    "LangGraph",
)

CONTEXTS = ("knowledge_base", "assistant")


def _walk_modules(package_name: str) -> Iterator[str]:
    package = importlib.import_module(package_name)
    yield package_name
    for info in pkgutil.walk_packages(package.__path__, prefix=f"{package_name}."):
        yield info.name


def _classes_defined_in(module_name: str) -> Iterator[tuple[str, type]]:
    module = importlib.import_module(module_name)
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module.__name__ and not name.startswith("_"):
            yield name, obj


def test_adapter_classes_carry_a_technology_prefix() -> None:
    violations: list[str] = []
    for context in CONTEXTS:
        for module_name in _walk_modules(f"knowledge_assistant.{context}.adapters"):
            if (
                ".http" in module_name
                or module_name.endswith("persistence.models")
                or module_name.endswith(".state")
            ):
                continue  # routers/schemas and ORM models are exempt by design
            for class_name, cls in _classes_defined_in(module_name):
                if issubclass(cls, pydantic.BaseModel):
                    continue  # adapter-local wire schemas (e.g. AnswerPayload)
                if not class_name.startswith(ADAPTER_PREFIXES):
                    violations.append(f"{module_name}.{class_name}")
    assert violations == [], "Infrastructure classes must carry a technology prefix: " + ", ".join(
        violations
    )


def test_port_modules_contain_only_protocols() -> None:
    violations: list[str] = []
    port_modules = [
        "knowledge_assistant.knowledge_base.application.ports",
        "knowledge_assistant.assistant.application.ports",
    ]
    for module_name in port_modules:
        for class_name, cls in _classes_defined_in(module_name):
            if not getattr(cls, "_is_protocol", False):
                violations.append(f"{module_name}.{class_name}")
    assert violations == [], f"Ports must be typing.Protocol classes: {', '.join(violations)}"
