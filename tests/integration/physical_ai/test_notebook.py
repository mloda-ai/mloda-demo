import importlib.util
import re
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

import pytest

from mloda_demo.physical_ai import style
from mloda_demo.physical_ai.readers import TumDepth

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "physical_ai.py"
CSS = NOTEBOOK.with_suffix(".css")
SLIDES = NOTEBOOK.parents[1] / "slides" / "physical_ai.html"


def _notebook() -> ModuleType:
    spec = importlib.util.spec_from_file_location("physical_ai_notebook", NOTEBOOK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_beat_runs_with_the_generic_reader() -> None:
    _, defs = _notebook().app.run()
    assert defs["chair"] == 53
    row = defs["at_frame"](defs["result"], defs["chair"])
    assert float(row["nearest_ahead_m"]) == pytest.approx(2.915, abs=0.005)
    assert not bool(row["stop"])
    assert defs["checked"].error == "DepthToMetres needs a declared scale; DepthPng delivers uint16 without one"


def test_the_tum_reader_stops_for_the_chair() -> None:
    _, defs = _notebook().app.run(defs={"reader": TumDepth})
    row = defs["at_frame"](defs["result"], defs["chair"])
    assert float(row["nearest_ahead_m"]) == pytest.approx(0.583, abs=0.005)
    assert bool(row["stop"])


def test_physical_ai_code_never_imports_torch() -> None:
    script = (
        "import runpy, sys\n"
        f"runpy.run_path({str(NOTEBOOK)!r}, run_name='__main__')\n"
        "assert 'mloda_demo.physical_ai.trace' in sys.modules\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)  # nosec B603
    assert result.returncode == 0, result.stderr[-2000:]


def _contrast(foreground: str, background: str) -> float:
    """WCAG 2 contrast ratio of two #RRGGBB colours."""

    def luminance(colour: str) -> float:
        channels = [int(colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_notebook_css_and_slides_work_offline_and_mirror_the_palette() -> None:
    assert f'css_file="{CSS.name}"' in NOTEBOOK.read_text()
    css = CSS.read_text()
    assert not any(remote in css + SLIDES.read_text() for remote in ("url(", "@import", "http"))
    tokens = dict(re.findall(r"--mloda-([a-z-]+): (#[0-9A-F]{6});", css))
    assert tokens == {
        "green": style.GREEN,
        "green-strong": style.GREEN_STRONG,
        "ink": style.INK,
        "muted": style.MUTED,
        "page": style.PAGE,
        "card": style.CARD,
        "panel": style.PANEL,
        "code": style.CODE,
        "on-dark": style.ON_DARK,
        "highlight": style.HIGHLIGHT,
        "red": style.RED,
        "red-strong": style.RED_STRONG,
    }


def test_palette_contrast_holds_for_text_and_marks() -> None:
    text = [
        (style.INK, style.PAGE),
        (style.MUTED, style.PAGE),
        (style.GREEN_STRONG, style.PAGE),
        (style.INK, style.CARD),
        (style.ON_DARK, style.CODE),
        (style.ON_DARK, style.PANEL),
        (style.INK, style.HIGHLIGHT),
        (style.RED_STRONG, style.CARD),
    ]
    marks = [(style.RED, style.PAGE), (style.GREEN, style.PAGE), (style.GREEN, style.PANEL)]
    assert min(_contrast(*pair) for pair in text) >= 4.5
    assert min(_contrast(*pair) for pair in marks) >= 3.0
