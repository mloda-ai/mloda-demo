import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from mloda_demo.retail.runs import Shown
from mloda_demo.retail.store import STORE
from tests.integration.conftest import assert_notebook_never_imports_torch, load_notebook

NOTEBOOKS = Path(__file__).resolve().parents[3] / "notebooks"
NOTEBOOK = NOTEBOOKS / "retail.py"
# The talk's closing slide: what the room knows, and the small change.
ROWS = (
    ("1 Find what exists", "The registry", "It is a package"),
    ("2 Understand a number", "The feature definition", "The definition itself runs"),
    ("3 Trust it", "Schema and validation", "A contract"),
    ("4 Use it", "Offline and online", "Declarative"),
    ("5 Ask something new", "On-demand features", "Nobody registered this combination"),
)


@pytest.fixture(autouse=True)
def empty_store() -> Iterator[None]:
    STORE.clear()
    yield
    STORE.clear()


def test_five_slides_each_a_header_and_one_command() -> None:
    cells = json.loads((NOTEBOOKS / "layouts" / "retail.slides.json").read_text())["data"]["cells"]
    assert cells[0] == {"type": "skip"}, "the setup cell stays off the slides"
    assert cells[1:] == [{"type": "slide"}, {"type": "fragment", "showCode": True}] * 5
    source = NOTEBOOK.read_text()
    for title, know, change in ROWS:
        assert f"## {title}" in source
        assert f"| {know} | {change} |" in source


def test_the_commands_are_plain_mloda_calls() -> None:
    source = NOTEBOOK.read_text()
    assert 'get_feature_group_docs(search="spend")' in source
    assert source.count("mloda.run_all(") == 4, "one call per slide from 2 to 5"
    assert "**" not in source, "every request carries its own configuration"
    assert "except FeatureResolutionError" in source


def test_every_command_shows_its_result() -> None:
    outputs, _ = load_notebook(NOTEBOOK).app.run()
    pages = [output._mime_()[1] for output in outputs if isinstance(output, Shown)]
    assert len(pages) == 5
    assert all(
        name in pages[0] for name in ("gross_spend", "net_spend", "net_spend_30d", "marketing", "finance", "risk")
    )
    assert "class NetSpend30d" in pages[1] and "customer 14045: 252.00" in pages[1]
    assert pages[1].index("Shared freely") < pages[1].index("Partly shared") < pages[1].index("def calculate_feature")
    assert "FeatureResolutionError: " in pages[2]
    assert "needs cancellations; the marketing export delivers orders only" in pages[2]
    assert "FeatureStore, customer 14045" in pages[3] and "NetSpend30d, customer 14045" in pages[3]
    assert "NetSpend30d, every customer" in pages[3] and "[85381 rows x 1 columns]" in pages[3]
    assert "<svg" in pages[4] and "customer 14045: 1,339.60" in pages[4]
    assert all(" ms" not in page for page in pages), "run times do not matter here"


def test_notebook_never_imports_torch() -> None:
    assert_notebook_never_imports_torch(NOTEBOOK, "mloda_demo.retail.runs")
