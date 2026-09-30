"""The plan as a picture, drawn from `mloda.explain` before any data moves: one box per step, coloured by owner."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from mloda.user import PlanStep

from mloda_demo.physical_ai.style import CARD, GREEN_STRONG, INK, MATPLOTLIB, MUTED
from mloda_demo.retail.sources import Orders, OrderSource

OWNER_COLOURS = {
    "finance": GREEN_STRONG,
    "marketing": "#B45309",
    "risk": "#1D4ED8",
    "logistics": "#6D28D9",
    "store": INK,
}


@dataclass(frozen=True)
class Node:
    label: str
    owner: str


def graph(steps: Sequence[PlanStep], source: type[OrderSource]) -> tuple[list[Node], list[tuple[int, int]]]:
    """Nodes for the compute steps, and edges from the step producing an input to the step reading it."""
    compute = [step for step in steps if step.step_kind == "compute" and step.feature_group is not None]
    nodes = [node(step, source) for step in compute]
    edges = {
        (index, target)
        for target, step in enumerate(compute)
        for name in step.input_feature_names
        for index, producer in enumerate(compute)
        if producer is not step and name in producer.feature_names
    }
    # The source feeds nearly every step; keep only its arrows that no other path already shows.
    roots = {index for index, step in enumerate(compute) if step.feature_group is Orders}
    return nodes, sorted(edge for edge in edges if edge[0] not in roots or not detour(edge, edges))


def detour(edge: tuple[int, int], edges: set[tuple[int, int]]) -> bool:
    """True when another path leads from the edge's start to its end."""
    start, end = edge
    frontier = [target for source, target in edges if source == start and target != end]
    seen: set[int] = set()
    while frontier:
        current = frontier.pop()
        if current == end:
            return True
        if current not in seen:
            seen.add(current)
            frontier.extend(target for source, target in edges if source == current)
    return False


def node(step: PlanStep, source: type[OrderSource]) -> Node:
    if step.feature_group is Orders:
        return Node(source.label.removeprefix("the "), source.owner)
    return Node(", ".join(sorted(step.feature_names)), str(getattr(step.feature_group, "OWNER", "")))


def depths(count: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Longest path from a root to each node."""
    depth = [0] * count
    for _ in range(count):
        for start, end in edges:
            depth[end] = max(depth[end], depth[start] + 1)
    return depth


@matplotlib.rc_context(MATPLOTLIB)
def draw(nodes: Sequence[Node], edges: Sequence[tuple[int, int]]) -> Figure:
    depth = depths(len(nodes), edges)
    columns = max(depth) + 1
    layers = [[index for index, d in enumerate(depth) if d == column] for column in range(columns)]
    rows = max(len(layer) for layer in layers)
    # A layer that an arrow jumps over sits lower, so the arrow passes above its boxes.
    jumped = {column for start, end in edges for column in range(depth[start] + 1, depth[end])}
    drop = 0.8 if jumped else 0.0
    width = max(2.8, 0.13 * max(len(n.label) for n in nodes) + 1.0)
    figure = Figure(figsize=(width * columns + 0.4, 1.6 * (rows + drop) + 0.4))
    axes = figure.add_subplot()
    axes.set_xlim(-0.5, columns - 0.5)
    axes.set_ylim(-0.5 - drop, rows - 0.5)
    axes.set_axis_off()
    position = {
        index: (column, (rows - 1) / 2 + (len(layer) - 1) / 2 - row - (drop if column in jumped else 0.0))
        for column, layer in enumerate(layers)
        for row, index in enumerate(layer)
    }
    for start, end in edges:
        (x0, y0), (x1, y1) = position[start], position[end]
        axes.annotate(
            "",
            xy=(x1 - 0.44, y1),
            xytext=(x0 + 0.44, y0),
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.6, "mutation_scale": 18},
        )
    for index, (x, y) in position.items():
        colour = OWNER_COLOURS.get(nodes[index].owner)
        axes.add_patch(
            FancyBboxPatch(
                (x - 0.42, y - 0.32),
                0.84,
                0.64,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=colour or CARD,
                edgecolor=colour or MUTED,
                linewidth=1.6,
            )
        )
        text = "white" if colour else INK
        axes.text(x, y + 0.07, nodes[index].label, ha="center", va="center", fontsize=14, color=text)
        axes.text(x, y - 0.2, nodes[index].owner, ha="center", va="center", fontsize=11, color=text)
    return figure


def svg(steps: Sequence[PlanStep], source: type[OrderSource]) -> str:
    buffer = StringIO()
    # A fresh salt keeps element ids unique when several pictures share one page.
    with matplotlib.rc_context({"svg.hashsalt": uuid.uuid4().hex}):
        draw(*graph(steps, source)).savefig(
            buffer, format="svg", transparent=True, bbox_inches="tight", metadata={"Date": None}
        )
    return buffer.getvalue()
