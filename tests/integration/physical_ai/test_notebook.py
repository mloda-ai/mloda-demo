import importlib.util
import re
import subprocess  # nosec B404
import sys
from pathlib import Path
from types import ModuleType

from mloda_demo.physical_ai import style
from mloda_demo.physical_ai.readers import DepthReaderB

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks" / "physical_ai.py"
CSS = NOTEBOOK.with_suffix(".css")
SLIDES = NOTEBOOK.parents[1] / "slides" / "physical_ai.html"


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


def test_the_fix_brings_every_column_back_to_brake() -> None:
    _, defs = _notebook().app.run(defs={"reader": DepthReaderB, "fault": False})
    table = defs["n_table"](defs["runs"])
    assert list(table.loc["object 7"]) == ["1.2 m", "1.2 m", "1.2 m"]
    assert list(table.loc["brake"]) == ["yes", "yes", "yes"]


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
    rest = re.sub(r'url\("data:font/woff2;base64,[A-Za-z0-9+/=]+"\)', "", css) + SLIDES.read_text()
    assert not any(remote in rest for remote in ("url(", "@import", "http"))
    assert re.findall(r'font-family: "([^"]+)";', css) == ["Schibsted Grotesk", "Bricolage Grotesque", "DM Mono"]
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
