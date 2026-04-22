"""Smoke test: the marimo notebook exports to a script and runs without error."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "demo" / "applydata_credit_xai.py"


@pytest.mark.slow
def test_notebook_runs_as_script(tmp_path: Path) -> None:
    assert NOTEBOOK.exists()
    script_path = tmp_path / "nb_script.py"
    subprocess.run(
        [sys.executable, "-m", "marimo", "export", "script", str(NOTEBOOK), "--output", str(script_path)],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Notebook script failed:\n{result.stderr}"
