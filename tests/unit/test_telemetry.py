import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from knowledge_assistant.platform.observability import telemetry
from knowledge_assistant.platform.observability.telemetry import (
    configure_telemetry,
    observe_operation,
    record_abstention,
    record_evidence,
    record_retry,
)


def test_disabled_telemetry_is_a_safe_noop() -> None:
    runtime = configure_telemetry(
        enabled=False,
        service_name="test-service",
        otlp_endpoint="http://collector:4318",
    )

    with observe_operation("retrieval", {"rag.retrieval.top_k": 5}):
        record_evidence(2)
        record_retry("retrieval")
        record_abstention()

    assert runtime.enabled is False
    runtime.shutdown()


def test_operation_observer_preserves_errors() -> None:
    with (
        pytest.raises(RuntimeError, match="provider failed"),
        observe_operation("generation"),
    ):
        raise RuntimeError("provider failed")


def test_enabled_operation_spans_never_export_exception_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        telemetry,
        "_tracer",
        lambda: provider.get_tracer("privacy-test"),
    )
    private_content = "private-client-deed.pdf: account 1234"

    with (
        pytest.raises(RuntimeError, match="private-client-deed"),
        observe_operation("extraction"),
    ):
        raise RuntimeError(private_content)

    [span] = exporter.get_finished_spans()
    exported = repr(
        {
            "attributes": span.attributes,
            "events": span.events,
            "status_description": span.status.description,
        }
    )
    assert span.status.status_code is StatusCode.ERROR
    attributes = span.attributes or {}
    assert attributes["error.type"] == "RuntimeError"
    assert span.events == ()
    assert private_content not in exported
    provider.shutdown()
