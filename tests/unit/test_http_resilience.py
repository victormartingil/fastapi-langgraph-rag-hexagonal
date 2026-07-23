"""Unit tests for the shared transient/permanent HTTP taxonomy.

The taxonomy decides WHAT gets retried across all three HTTP-facing adapters:
retrying a permanent error wastes time and hides misconfiguration; NOT
retrying a transient one turns a blip into an outage. These tests pin the
classification itself (adapter-level retry behavior is covered in
test_ai_adapters.py).
"""

import httpx

from knowledge_assistant.platform.http.resilience import (
    is_transient_http_error,
)


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://provider.test/api")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


class TestTransientTaxonomy:
    def test_429_rate_limit_is_transient(self) -> None:
        # 429 is the canonical "slow down and try again" — always retryable.
        assert is_transient_http_error(_status_error(429))

    def test_server_dropping_the_connection_mid_flight_is_transient(self) -> None:
        # RemoteProtocolError/ReadError: the server accepted the connection
        # and vanished — a blip, not a rejection.
        assert is_transient_http_error(httpx.RemoteProtocolError("disconnected"))
        assert is_transient_http_error(httpx.ReadError("connection reset"))

    def test_interrupted_writes_and_half_closed_sockets_are_transient(self) -> None:
        # WriteError/CloseError: the request may never have reached the
        # server, and the adapters' POSTs are idempotent — retry is safe.
        assert is_transient_http_error(httpx.WriteError("broken pipe"))
        assert is_transient_http_error(httpx.CloseError("connection closed"))

    def test_egress_proxy_failures_are_transient(self) -> None:
        # ProxyError is the LOCAL proxy failing: the provider never saw the
        # request, so blaming it (or giving up) would both be wrong.
        assert is_transient_http_error(httpx.ProxyError("proxy tunnel failed"))

    def test_5xx_is_transient(self) -> None:
        assert is_transient_http_error(_status_error(500))
        assert is_transient_http_error(_status_error(503))

    def test_client_errors_are_permanent(self) -> None:
        # 401/403/404/422 can never succeed on retry — they are configuration
        # or request bugs and must fail fast.
        for status_code in (400, 401, 403, 404, 422):
            assert not is_transient_http_error(_status_error(status_code))
