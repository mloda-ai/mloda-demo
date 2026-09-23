"""Top-down view: camera at the origin, fixed axes, the brake circle."""

from __future__ import annotations

import math

import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from mloda_demo.physical_ai.definition import BRAKE_THRESHOLD_M

RANGE_M = 13.0
BRAKE_COLOUR = "#EF4444"
CLEAR_COLOUR = "#23A455"


def top_down(points: pd.DataFrame, *, ghost: tuple[float, float] | None = None, title: str = "") -> Figure:
    """Objects at their nearest point (x forward, y left); objects beyond the axes sit on the edge with their distance."""
    figure = Figure(figsize=(5, 5))
    axes = figure.subplots()
    axes.add_patch(Circle((0.0, 0.0), BRAKE_THRESHOLD_M, color=BRAKE_COLOUR, alpha=0.12))
    axes.plot(0.0, 0.0, marker="^", color="black", markersize=10, clip_on=False)
    if ghost is not None:
        axes.scatter(-ghost[1], ghost[0], s=160, facecolors="none", edgecolors="grey", alpha=0.6, linestyles="--")
    for object_id, point in points.iterrows():
        across, forward = -float(point["y_m"]), float(point["x_m"])
        colour = BRAKE_COLOUR if point["brake"] else CLEAR_COLOUR
        distance = math.hypot(across, forward)
        scale = min(RANGE_M / max(forward, 1e-9), (RANGE_M / 2) / max(abs(across), 1e-9))
        if scale >= 1.0:
            axes.scatter(across, forward, s=160, color=colour, zorder=3)
            axes.annotate(str(object_id), (across, forward), xytext=(8, 4), textcoords="offset points")
        else:
            edge = (across * scale * 0.985, forward * scale * 0.985)
            axes.scatter(*edge, s=80, marker="^", facecolors="none", edgecolors=colour, zorder=3, clip_on=False)
            axes.annotate(
                f"{object_id}: {distance:.0f} m",
                edge,
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                annotation_clip=False,
            )
    axes.set_xlim(-RANGE_M / 2, RANGE_M / 2)
    axes.set_ylim(0.0, RANGE_M)
    axes.set_aspect("equal")
    axes.set_xlabel("left / right (m)")
    axes.set_ylabel("forward (m)")
    axes.set_title(title, pad=20)
    return figure
