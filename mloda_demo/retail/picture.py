"""The plan as a picture, drawn from `mloda.explain` before any data moves: one outlined box per step, its owner in grey."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from itertools import pairwise

import matplotlib
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from mloda.user import PlanStep

from mloda_demo.physical_ai.style import INK, MATPLOTLIB, MUTED
from mloda_demo.retail.sources import Orders, OrderSource

LABEL_SIZE, OWNER_SIZE = 18, 13  # points
CHARACTER, LINE = 0.14, 0.34  # inches per character and per line of a label
GAP_X, GAP_Y, EDGE = 0.6, 0.35, 0.1  # inches between columns, between rows, around the picture


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


def lines_of(label: str) -> list[str]:
    """One line per name; a chained name breaks at every double underscore, one line per part."""
    lines: list[str] = []
    for name in label.split(", "):
        breaks = [0, *(index for index in range(1, len(name)) if name.startswith("__", index)), len(name)]
        lines += [name[start:end] for start, end in pairwise(breaks)]
    return lines


@matplotlib.rc_context(MATPLOTLIB)
def draw(nodes: Sequence[Node], edges: Sequence[tuple[int, int]]) -> Figure:
    """Boxes in columns by depth, measured in inches so the text keeps its size next to them."""
    depth = depths(len(nodes), edges)
    layers = [[index for index, d in enumerate(depth) if d == column] for column in range(max(depth) + 1)]
    rows = max(len(layer) for layer in layers)
    labels = [lines_of(node.label) for node in nodes]
    widths = [
        max(1.9, CHARACTER * max(len(line) for index in layer for line in labels[index]) + 0.5) for layer in layers
    ]
    lefts = [EDGE + sum(widths[:column]) + column * GAP_X for column in range(len(layers))]
    height = LINE * max(len(lines) for lines in labels) + 0.7
    # A layer that an arrow jumps over sits lower, so the arrow passes above its boxes.
    jumped = {column for start, end in edges for column in range(depth[start] + 1, depth[end])}
    drop = height / 2 + 0.25 if jumped else 0.0
    down = height + GAP_Y
    size = (lefts[-1] + widths[-1] + EDGE, rows * down - GAP_Y + drop + 2 * EDGE)
    figure = Figure(figsize=size)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, size[0])
    axes.set_ylim(0, size[1])
    axes.set_axis_off()
    position = {
        index: (
            lefts[column] + widths[column] / 2,
            size[1] - EDGE - height / 2 - ((rows - len(layer)) / 2 + row) * down - (drop if column in jumped else 0.0),
        )
        for column, layer in enumerate(layers)
        for row, index in enumerate(layer)
    }
    for start, end in edges:
        (x0, y0), (x1, y1) = position[start], position[end]
        axes.annotate(
            "",
            xy=(x1 - widths[depth[end]] / 2 - 0.04, y1),
            xytext=(x0 + widths[depth[start]] / 2 + 0.04, y0),
            arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.8, "mutation_scale": 22},
        )
    for index, (x, y) in position.items():
        width = widths[depth[index]]
        axes.add_patch(
            FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0,rounding_size=0.12",
                facecolor="white",
                edgecolor=INK,
                linewidth=1.2,
            )
        )
        lines = labels[index]
        for line_number, line in enumerate(lines):
            above = (len(lines) - 1) / 2 - line_number
            axes.text(x, y + 0.17 + above * LINE, line, ha="center", va="center", fontsize=LABEL_SIZE, color=INK)
        axes.text(
            x, y - height / 2 + 0.24, nodes[index].owner, ha="center", va="center", fontsize=OWNER_SIZE, color=MUTED
        )
    return figure


def svg(steps: Sequence[PlanStep], source: type[OrderSource]) -> str:
    buffer = StringIO()
    # A fresh salt keeps element ids unique when several pictures share one page.
    with matplotlib.rc_context({"svg.hashsalt": uuid.uuid4().hex}):
        draw(*graph(steps, source)).savefig(
            buffer, format="svg", transparent=True, bbox_inches="tight", metadata={"Date": None}
        )
    return buffer.getvalue()
