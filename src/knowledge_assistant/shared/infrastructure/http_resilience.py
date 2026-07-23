"""Shared HTTP resilience policy for the AI adapters.

Three adapters (two embedding providers, one LLM generator) call remote
model APIs and retry with tenacity. WHAT is retried matters as much as the
backoff: retrying a 401 or 404 wastes seconds and rate-limit budget on a
request that will never succeed, and it hides configuration bugs behind
"flaky service" symptoms. So the policy lives in ONE place:

- transient (retry): timeouts, connection errors, the server (or an egress
  proxy) dropping the connection mid-flight — RemoteProtocolError,
  ReadError, WriteError, CloseError, ProxyError — 5xx responses, and 429
  (rate limited — the canonical "slow down and try again" signal);
- permanent (fail fast): everything else, including all other 4xx.

WriteError/CloseError are worth classifying consciously: the request may
never have reached the model server, and the adapter POSTs are idempotent,
so a retry cannot corrupt anything. ProxyError is the LOCAL egress proxy
failing — the provider never even saw us.

Retry-After note: honoring the server's `Retry-After` header on 429 would
require a custom tenacity wait strategy inspecting each attempt's response
— deliberately NOT done here (the exponential backoff with an 8s cap is a
reasonable approximation); revisited if a provider starts rate-limiting in
practice.
"""

import httpx


def is_transient_http_error(exc: BaseException) -> bool:
    """True iff the failed call is worth retrying (tenacity predicate)."""
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError):
        return True
    if isinstance(
        exc,
        httpx.RemoteProtocolError
        | httpx.ReadError
        | httpx.WriteError
        | httpx.CloseError
        | httpx.ProxyError,
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status >= 500 or status == 429
    return False
