"""Architecture tests, part 1: import-linter contracts.

The contracts live in pyproject.toml (`[tool.importlinter]`) and encode the
hexagonal rules as executable checks:

- within each context: infrastructure -> application -> domain (inward only);
- contexts must not import each other;
- the domain must not import any framework (fastapi, sqlalchemy, langgraph...).

If these pass, the architecture has not eroded — no matter who (or which AI)
wrote the latest PR. No Docker needed.
"""

import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_import_linter_contracts_hold() -> None:
    lint_imports = shutil.which("lint-imports")
    assert lint_imports is not None, "lint-imports not found; run via `uv run pytest`"
    result = subprocess.run(
        [lint_imports],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Architecture contracts violated:\n{result.stdout}\n{result.stderr}"
    )
