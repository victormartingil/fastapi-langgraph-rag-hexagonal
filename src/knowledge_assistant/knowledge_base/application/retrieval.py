"""Retrieval application concepts that are not ports."""

from enum import StrEnum


class RetrievalStrategy(StrEnum):
    """Supported production retrieval modes.

    The assistant uses `HYBRID`; the evaluator can run the same PostgreSQL
    adapter in `DENSE` or `LEXICAL` mode to make quality trade-offs explicit.
    """

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
