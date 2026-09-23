"""mloda.ai design tokens for the Physical AI plots; `notebooks/physical_ai.css` mirrors the colours."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import Any

from matplotlib import font_manager

# Static 400 instance of the mloda.ai woff2 (fontTools varLib.instancer): matplotlib cannot read woff2.
FONTS = Path(__file__).parent / "fonts"

GREEN = "#23A455"
GREEN_STRONG = "#12813C"  # green text on white, and white text on green
INK = "#0E120F"
MUTED = "#6A6C6A"  # ink at 62% on white
PAGE = "#FFFFFF"
CARD = "#F5F7F9"
PANEL = "#0F1611"
CODE = "#18201A"
ON_DARK = "#F4F7F4"
HIGHLIGHT = "#FAD689"
RED = "#EF4444"
RED_STRONG = "#B91C1C"  # red text

# Any keys: matplotlib 3.11 types rc keys as literals, 3.10 as str.
MATPLOTLIB: dict[Any, Any] = {
    "font.family": "Schibsted Grotesk",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.titlesize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
}


@cache
def register_fonts() -> None:
    for path in sorted(FONTS.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
