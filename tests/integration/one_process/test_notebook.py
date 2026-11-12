from pathlib import Path

from mloda_demo.one_process import Lineage
from tests.integration.conftest import assert_notebook_never_imports_torch, load_notebook

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "one_process.py"


def test_every_beat_draws_its_picture():
    outputs, _ = load_notebook(NOTEBOOK).app.run()
    pictures = [output for output in outputs if isinstance(output, Lineage)]
    assert [len(picture.features) for picture in pictures] == [2, 2, 2, 2, 3]
    assert all(picture.title for picture in pictures), "a beat without a title makes a slide without a heading"
    receipts = sum(getattr(output, "text", "").count('<pre class="trace">') for output in outputs)
    assert receipts == 2, "the receipt slide shows the generic PNG run and the TUM run"


def test_notebook_never_imports_torch():
    assert_notebook_never_imports_torch(NOTEBOOK, "mloda_demo.one_process.lineage")
