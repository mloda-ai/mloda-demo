import importlib.util
import json
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mloda_demo.retail.runs import Shown
from mloda_demo.runtime_lineage.telemetry import LINEAGE_URL, OTLP_ENDPOINT

NOTEBOOKS = Path(__file__).resolve().parents[3] / "notebooks"
NOTEBOOK = NOTEBOOKS / "runtime_lineage.py"
ROWS = (
    ("1 One answer, one ID", "A trace", "Every span is a data step with a lineage twin"),
    ("2 Another request, another path", "Lineage as a fixed map", "It belongs to the request"),
    ("3 The wrong number", "Red spans", "A green span carries evidence"),
    ("4 Make it fail loudly", "Alerts on failures", "Wrong data became a failure"),
)
SLIDE, COMMAND = {"type": "slide"}, {"type": "fragment", "showCode": True}


@pytest.fixture(autouse=True)
def no_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    """The notebook reads the stack's addresses from the environment; the tests export nothing."""
    monkeypatch.delenv(OTLP_ENDPOINT, raising=False)
    monkeypatch.delenv(LINEAGE_URL, raising=False)


def _notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("runtime_lineage_notebook", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_case_and_four_steps_each_a_header_and_its_commands() -> None:
    cells = json.loads((NOTEBOOKS / "layouts" / "runtime_lineage.slides.json").read_text())["data"]["cells"]
    assert cells[0] == {"type": "skip"}, "the setup cell stays off the slides"
    # Step 4 has two commands: the refused export, then the office with the right chart.
    assert cells[1:] == [SLIDE, COMMAND] * 5 + [COMMAND]
    source = NOTEBOOK.read_text()
    assert "## The case" in source
    for title, know, change in ROWS:
        assert f"## {title}" in source
        assert f"| {know} | {change} |" in source


def test_the_commands_are_plain_mloda_calls_with_the_hooks() -> None:
    source = NOTEBOOK.read_text()
    assert source.count("mloda.run_all(") == source.count("function_extender=hooks") == 6
    # The chart the office reads is on every slide; only the last command changes it.
    assert source.count("plugin_collector=reads_skr04") == 5 and source.count("plugin_collector=reads_skr03") == 1
    assert "except ChartConflict" in source


def test_every_command_shows_its_result() -> None:
    outputs, _ = _notebook().app.run()
    pages = [output._mime_()[1] for output in outputs if isinstance(output, Shown)]
    assert len(pages) == 6
    assert "13,750.50" in pages[0] and "4 steps, all ok" in pages[0] and "<table" not in pages[0]
    assert "revenue__sources, 22,485.06" in pages[1] and "<td>DatevExtfReader</td>" in pages[1]
    assert "revenue__sources, 22,485.06" in pages[2] and "<td>GdpduReader</td>" in pages[2]
    assert "13,750.50" in pages[3] and "chart SKR04" in pages[3] and "<td>error</td>" not in pages[3]
    assert "ChartConflict: " in pages[4] and "<td>error</td>" in pages[4]
    assert "receivables__sources, 7,500.00" in pages[5] and "chart SKR03" in pages[5]
    for number, page in enumerate(pages[1:], start=1):
        linked = "3 of 3" if number == 4 else "4 of 4"  # the refused run ends at its third step
        assert f"{linked} lineage steps point at this trace" in page and "<svg" in page


def test_notebook_never_imports_torch() -> None:
    script = (
        "import runpy, sys\n"
        f"runpy.run_path({str(NOTEBOOK)!r}, run_name='__main__')\n"
        "assert 'mloda_demo.runtime_lineage.show' in sys.modules\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # nosec B603
    assert result.returncode == 0, result.stderr[-2000:]
