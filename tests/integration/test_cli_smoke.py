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
    run = subprocess.run(
        [cli, "run", "credit_risk", "--customer", "app-customer-c"], capture_output=True, text=True, check=True
    )
    predicted, joined = (dict(zip(*[line.split() for line in r.stdout.splitlines()])) for r in (result, run))
    assert predicted["credit_risk"] == joined["credit_risk"]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["age", "credit_amount", "--filter", "age,range,20:40"],
            {"customer_id": "app-customer-e", "age": "33", "credit_amount": "3000"},
        ),
        (
            ["credit_risk", "duration", "--filter", "age,range,20:40"],
            {"customer_id": "app-customer-e", "duration": "24"},
        ),
        (["credit_amount", "--filter", "age,range,20:40"], {"customer_id": "app-customer-e", "credit_amount": "3000"}),
        (
            ["credit_amount", "--filter", "credit_amount,equal,3000"],
            {"customer_id": "app-customer-e", "credit_amount": "3000"},
        ),
        (
            ["credit_amount", "age", "--filter", "customer_id,equal,app-customer-b"],
            {"customer_id": "app-customer-b", "age": "47"},
        ),
        (
            ["duration", "credit_amount", "--customer", "app-customer-c"],
            {"customer_id": "app-customer-c", "duration": "48", "credit_amount": "1200"},
        ),
    ],
)
def test_cli_run_keeps_customer_ids_on_their_rows(args: list[str], expected: dict[str, str]) -> None:
    cli = _resolve_cli()
    result = subprocess.run([cli, "run", *args], capture_output=True, text=True, check=True)
    header, *rows = [line.split() for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert expected.items() <= dict(zip(header, rows[0])).items()


def test_cli_run_credit_risk_with_no_matching_rows_exits_cleanly() -> None:
    cli = _resolve_cli()
    result = subprocess.run(
        [cli, "run", "credit_risk", "--filter", "age,range,90:99"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 1
    assert "rows match" in result.stderr
    assert "Traceback" not in result.stderr


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
