"""Unit tests for the FTS-language parity comparison (ADR-0004).

The pure half of the startup guard: given the language recorded in
schema_meta and the configured one, pass on equality and fail LOUDLY on
mismatch — naming both languages and the fix. The I/O half (reading the
table, distinguishing a missing table from an unreachable database) is
covered by integration tests against a real migrated database.
"""

import pytest
from sqlalchemy.exc import ProgrammingError

from knowledge_assistant.platform.database.schema_meta import (
    _assert_language_matches,
    _is_missing_meta_table,
)


class TestFtsLanguageParity:
    def test_matching_languages_pass(self) -> None:
        _assert_language_matches("english", "english")

    def test_mismatch_fails_fast_naming_both_languages_and_the_fix(self) -> None:
        with pytest.raises(ValueError, match="FTS language mismatch") as exc_info:
            _assert_language_matches("spanish", "english")

        message = str(exc_info.value)
        assert "'spanish'" in message  # what the schema was built with
        assert "'english'" in message  # what the app is configured for
        assert "alembic upgrade head" in message  # the way out


def _programming_error_with_sqlstate(sqlstate: str | None) -> ProgrammingError:
    """Build a ProgrammingError whose driver original carries a sqlstate,
    mimicking asyncpg's PostgresError (which exposes the code as .sqlstate)."""
    orig = Exception("driver error")
    if sqlstate is not None:
        orig.sqlstate = sqlstate  # type: ignore[attr-defined]
    return ProgrammingError("SELECT ...", {}, orig)


class TestMissingTableClassification:
    def test_undefined_table_is_recognized(self) -> None:
        # pgcode 42P01: the ONLY ProgrammingError the guard may reinterpret
        # as 'run the migrations'.
        assert _is_missing_meta_table(_programming_error_with_sqlstate("42P01"))

    def test_insufficient_privilege_is_not_a_missing_table(self) -> None:
        # pgcode 42501 must surface honestly — telling the operator to 'run
        # alembic upgrade head' for a permissions problem would be a lie.
        assert not _is_missing_meta_table(_programming_error_with_sqlstate("42501"))

    def test_errors_without_a_sqlstate_are_not_a_missing_table(self) -> None:
        assert not _is_missing_meta_table(_programming_error_with_sqlstate(None))
