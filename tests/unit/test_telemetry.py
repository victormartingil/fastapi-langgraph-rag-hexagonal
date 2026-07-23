import pytest

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
