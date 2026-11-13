"""Integration: exercise the mloda-demo CLI entry point as a real subprocess.

No LLM, no claude. We're proving the console_scripts entry resolves and that
the demo's commands run end-to-end against the real FGs and model artifact.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.slow


def _resolve_cli() -> str:
    path = shutil.which("mloda-demo")
    if path is None:
        pytest.skip("mloda-demo console script not on PATH (run `uv sync --all-extras`).")
    return path


def test_cli_discover_lists_all_demo_fgs() -> None:
    cli = _resolve_cli()
    result = subprocess.run([cli, "discover"], capture_output=True, text=True, check=True)
    for fg in (
        "ApplicationsFG",
        "FinancialsFG",
        "QuestionnaireFG",
        "CreditRiskClassifierFG",
        "ZennitAttributionFeatureGroup",
        "GradientAttributionFeatureGroup",
    ):
        assert fg in result.stdout, f"{fg} missing from discover output"


def test_cli_predict_returns_verdict_for_known_customer() -> None:
    cli = _resolve_cli()
    result = subprocess.run(
        [cli, "predict", "--customer", "app-customer-c"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "app-customer-c" in result.stdout
    assert any(verdict in result.stdout for verdict in ("good", "bad"))


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (["age", "credit_amount"], ("3000", "33")),
        (["credit_risk", "duration"], ("24",)),
        (["credit_amount"], ("3000",)),
    ],
)
def test_cli_run_filter_keeps_customer_ids_on_their_rows(features: list[str], expected: tuple[str, ...]) -> None:
    cli = _resolve_cli()
    result = subprocess.run(
        [cli, "run", *features, "--filter", "age,range,20:40"],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = [line.split() for line in result.stdout.splitlines()[1:]]
    assert len(rows) == 1
    assert rows[0][0] == "app-customer-e"
    assert set(expected) <= set(rows[0])


def test_cli_plan_credit_risk_joins_the_input_fgs() -> None:
    cli = _resolve_cli()
    result = subprocess.run([cli, "plan", "credit_risk"], capture_output=True, text=True, check=True)
    assert "CreditRiskClassifierFG" in result.stdout
    assert "joined" in result.stdout


def test_cli_explain_runs_for_default_method() -> None:
    cli = _resolve_cli()
    result = subprocess.run(
        [cli, "explain", "--customer", "app-customer-c"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "app-customer-c" in result.stdout
    assert "method=EpsilonPlus" in result.stdout
