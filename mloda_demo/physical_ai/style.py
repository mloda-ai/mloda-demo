"""mloda.ai colours for the Physical AI plots; `notebooks/physical_ai.css` mirrors them."""

from __future__ import annotations

from typing import Any

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
