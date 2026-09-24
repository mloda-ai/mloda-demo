"""The robot's frame (camera, depth map, the number) and pipeline pictures drawn from traces."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from numpy.typing import NDArray

from mloda_demo.physical_ai.clip import corridor
from mloda_demo.physical_ai.style import CARD, GREEN, GREEN_STRONG, INK, MATPLOTLIB, MUTED, RED_STRONG

ATTRIBUTION = "TUM RGB-D benchmark, freiburg2_pioneer_slam (CC BY 4.0), Sturm et al., IROS 2012"


@matplotlib.rc_context(MATPLOTLIB)
def frame_view(rgb: NDArray[np.uint8], depth_m: NDArray[np.float64], nearest: float, stop: bool, t_s: float) -> Figure:
    """Camera large on the left, the depth map as any viewer shows it (auto-scaled) on the right, the number below."""
    figure = Figure(figsize=(11, 4.9), dpi=72)
    grid = figure.add_gridspec(1, 2, width_ratios=[3, 2], left=0.02, right=0.98, top=0.9, bottom=0.2, wspace=0.05)
    camera = figure.add_subplot(grid[0, 0])
    camera.imshow(rgb)
    camera.set_axis_off()
    camera.set_title(f"the robot's camera, t = {t_s:.1f} s", fontsize=12, color=MUTED, loc="left")
    depth = figure.add_subplot(grid[0, 1])
    depth.imshow(depth_m, cmap="viridis")
    depth.set_axis_off()
    depth.set_title("depth, auto-scaled", fontsize=12, color=MUTED, loc="left")
    band = corridor(depth_m)
    rows, columns = depth_m.shape
    depth.add_patch(
        Rectangle(
            (columns // 3, rows // 4), band.shape[1], band.shape[0], fill=False, edgecolor="white", linestyle="--"
        )
    )
    verdict, colour = ("STOP", RED_STRONG) if stop else ("go", GREEN_STRONG)
    figure.text(0.02, 0.07, f"nearest ahead: {nearest:.2f} m", fontsize=24, fontweight="bold", color=INK)
    figure.text(0.62, 0.07, verdict, fontsize=24, fontweight="bold", color=colour)
    figure.text(0.98, 0.01, ATTRIBUTION, fontsize=7, color=MUTED, ha="right")
    return figure


@matplotlib.rc_context(MATPLOTLIB)
def pipeline_graph(chains: Sequence[Sequence[str]]) -> Figure:
    """Boxes and arrows, one row per chain; a step every chain has is drawn once, after each chain's own steps."""
    shared = [step for step in chains[0] if all(step in chain for chain in chains[1:])]
    own = [[step for step in chain if step not in shared] for chain in chains]
    if any(not steps for steps in own):
        raise ValueError("every chain needs a step of its own before the shared ones")
    rows = len(chains)
    columns = max(len(chain) for chain in chains)
    figure = Figure(figsize=(2.6 * columns + 0.6, 1.5 * rows + 0.6))
    axes = figure.add_subplot()
    axes.set_xlim(-0.5, columns - 0.5)
    axes.set_ylim(-0.5, rows - 0.5)
    axes.set_axis_off()
    positions: dict[tuple[int, str], tuple[float, float]] = {}

    def box(x: float, y: float, label: str, *, accent: bool) -> None:
        axes.add_patch(
            FancyBboxPatch(
                (x - 0.42, y - 0.3),
                0.84,
                0.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=GREEN if accent else CARD,
                edgecolor=GREEN_STRONG if accent else MUTED,
                linewidth=1.6,
            )
        )
        axes.text(x, y, label, ha="center", va="center", fontsize=15, color="white" if accent else INK)

    def arrow(start: tuple[float, float], end: tuple[float, float]) -> None:
        axes.annotate(
            "",
            xy=(end[0] - 0.44, end[1]),
            xytext=(start[0] + 0.44, start[1]),
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.6, "mutation_scale": 18},
        )

    centre = (rows - 1) / 2
    for index, chain in enumerate(chains):
        y = rows - 1 - index
        for column, step in enumerate(own[index]):
            box(column, y, step, accent=column == 0)
            positions[(index, step)] = (column, y)
            if column:
                arrow(positions[(index, own[index][column - 1])], (column, y))
    for column, step in enumerate(shared):
        x = column + max(len(steps) for steps in own)
        box(x, centre, step, accent=False)
        for index, steps in enumerate(own):
            previous = positions[(index, steps[-1])] if column == 0 else (x - 1, centre)
            arrow(previous, (x, centre))
    return figure
