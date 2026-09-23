import importlib.util
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

from mloda_demo.physical_ai.readers import DepthReaderB

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "physical_ai.py"


def _notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("physical_ai_notebook", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_beat_runs_with_the_new_vendor_line_changed() -> None:
    _, defs = _notebook().app.run(defs={"reader": DepthReaderB})
    n_table = defs["n_table"]
    assert list(n_table(defs["runs"]).loc["brake"]) == ["yes", "yes", "NO"]
    assert list(n_table(defs["asked"]).loc["approaching"]) == ["3", "3", "3"]
    assert defs["checked"].error == "DepthToMetres assumes uint16 depth in cm, but DepthReaderB delivers mm"


def test_physical_ai_code_never_imports_torch() -> None:
    script = (
        "import runpy, sys\n"
        f"runpy.run_path({str(NOTEBOOK)!r}, run_name='__main__')\n"
        "assert 'mloda_demo.physical_ai.trace' in sys.modules\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # nosec B603
    assert result.returncode == 0, result.stderr[-2000:]
