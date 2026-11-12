import importlib.util
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType


def load_notebook(notebook: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{notebook.stem}_notebook", notebook)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_notebook_never_imports_torch(notebook: Path, expected_module: str) -> None:
    script = (
        "import runpy, sys\n"
        f"runpy.run_path({str(notebook)!r}, run_name='__main__')\n"
        f"assert {expected_module!r} in sys.modules\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # nosec B603
    assert result.returncode == 0, result.stderr[-2000:]
