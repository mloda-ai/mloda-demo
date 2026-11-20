import json
from pathlib import Path

import pytest
from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.community.extenders.otel.otel_extender import OtelExtender

from mloda_demo.retail.runs import Shown
from mloda_demo.runtime_lineage.telemetry import LINEAGE_URL, OTLP_ENDPOINT
from tests.integration.conftest import assert_notebook_never_imports_torch, load_notebook

NOTEBOOKS = Path(__file__).resolve().parents[3] / "notebooks"
NOTEBOOK = NOTEBOOKS / "runtime_lineage.py"
TICKET = (
    "| Question | Client says | Monitoring |",
    "| Open receivables of client B? | 7,500.00 | all green |",
    "The number goes into the 2025 closing.",
)
HOOKS = "hooks = {OtelExtender(tracer_provider=traces), OpenLineageExtender(client=lineage), *metrics}"
CASE = "The case: two numbers, no suspect"
STEPS = ("The witness: what the hooks recorded", "The suspect: the chart of accounts", "The fix: make it fail loudly")
SLIDE, COMMAND = {"type": "slide"}, {"type": "fragment", "showCode": True}


@pytest.fixture(autouse=True)
def no_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """The notebook reads the stack's addresses from the environment; the tests export nothing."""
    monkeypatch.delenv(OTLP_ENDPOINT, raising=False)
    monkeypatch.delenv(LINEAGE_URL, raising=False)


def test_the_case_and_three_steps_each_a_header_and_its_command() -> None:
    cells = json.loads((NOTEBOOKS / "layouts" / "runtime_lineage.slides.json").read_text())["data"]["cells"]
    assert cells[0] == {"type": "skip"}, "the setup cell stays off the slides"
    assert cells[1:] == [SLIDE, COMMAND] * 4
    source = NOTEBOOK.read_text()
    # The first screen: the ticket, then the two hooks, then the call.
    case = source[source.index(f"## {CASE}") : source.index("mloda.run_all(")]
    places = [case.find(line) for line in (*TICKET, HOOKS)]
    assert -1 not in places and places == sorted(places)
    # A step's header is its title, nothing else.
    for title in STEPS:
        assert f'mo.md(r"""\n    ## {title}\n    """)' in source


def test_the_commands_are_plain_mloda_calls_with_the_hooks() -> None:
    source = NOTEBOOK.read_text()
    assert source.count("mloda.run_all(") == source.count("function_extender=hooks") == 4
    # The chart the office reads is on every slide; only the suspect's command reads with the other one.
    assert source.count("plugin_collector=reads_skr04") == 3 and source.count("plugin_collector=reads_skr03") == 1
    assert "except ChartConflict" in source


def test_every_command_shows_its_result() -> None:
    outputs, defs = load_notebook(NOTEBOOK).app.run()
    assert {type(hook) for hook in defs["hooks"]} == {OtelExtender, OpenLineageExtender}
    pages = [output._mime_()[1] for output in outputs if isinstance(output, Shown)]
    assert len(pages) == 4
    # The case shows what a dashboard would: the number and how the run ended.
    assert "13,750.50" in pages[0] and "4 steps, all ok" in pages[0] and "<table" not in pages[0]
    # One ticket: the answer opened, the same file read with the other chart, then the refusal.
    assert "13,750.50" in pages[1] and "chart SKR04" in pages[1] and "<td>error</td>" not in pages[1]
    assert "receivables__sources, 7,500.00" in pages[2] and "chart SKR03" in pages[2]
    same_file = pages[1].partition("file datev:client_b@")[2][:12]
    assert same_file and f"file datev:client_b@{same_file}" in pages[2]
    assert "ChartConflict: " in pages[3] and "<td>error</td>" in pages[3]
    for number, page in enumerate(pages[1:], start=1):
        linked = "3 of 3" if number == 3 else "4 of 4"  # the refused run ends at its third step
        assert f"{linked} lineage steps point at this trace" in page and "<svg" in page


def test_notebook_never_imports_torch() -> None:
    assert_notebook_never_imports_torch(NOTEBOOK, "mloda_demo.runtime_lineage.show")
