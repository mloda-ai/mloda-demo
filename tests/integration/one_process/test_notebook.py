import importlib.util
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

from mloda_demo.one_process import Lineage

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "one_process.py"


def _notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("one_process_notebook", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_beat_draws_its_picture():
    outputs, _ = _notebook().app.run()
    pictures = [output for output in outputs if isinstance(output, Lineage)]
    assert [len(picture.features) for picture in pictures] == [2, 2, 2, 2, 3]
    assert all(picture.title for picture in pictures), "a beat without a title makes a slide without a heading"
    receipts = sum(getattr(output, "text", "").count('<pre class="trace">') for output in outputs)
    assert receipts == 2, "the receipt slide shows the generic PNG run and the TUM run"


def test_notebook_never_imports_torch():
    script = (
        "import runpy, sys\n"
        f"runpy.run_path({str(NOTEBOOK)!r}, run_name='__main__')\n"
        "assert 'mloda_demo.one_process.lineage' in sys.modules\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # nosec B603
    assert result.returncode == 0, result.stderr[-2000:]
