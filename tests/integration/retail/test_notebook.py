import importlib.util
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

from mloda_demo.retail.agent import Slide

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "retail.py"


def _notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("retail_notebook", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_step_renders_a_slide():
    outputs, _ = _notebook().app.run()
    slides = [output for output in outputs if isinstance(output, Slide)]
    assert [slide.step for slide in slides if slide.step] == [
        "The support agent",
        "1 · Find what exists",
        "2 · Understand a number",
        "3 · Trust it",
        "4 · Use it: offline, online, agent",
        "5 · Ask something new",
    ]
    assert len(slides) == 8, "org, checkout, the opening question and the five steps"


def test_notebook_never_imports_torch():
    script = (
        "import runpy, sys\n"
        f"runpy.run_path({str(NOTEBOOK)!r}, run_name='__main__')\n"
        "assert 'mloda_demo.retail.agent' in sys.modules\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # nosec B603
    assert result.returncode == 0, result.stderr[-2000:]
