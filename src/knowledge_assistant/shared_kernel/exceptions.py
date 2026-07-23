"""Shared domain exceptions.

Every domain error in the system inherits from ``DomainError``. This gives the
HTTP layer a single, stable seam: the error handler maps subclasses of
``DomainError`` to HTTP responses without knowing anything else about the
domain. Domain code, in turn, never imports HTTP concepts.
"""


class DomainError(Exception):
    """Base class for all business-rule violations in the system."""
