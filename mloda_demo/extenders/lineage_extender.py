"""Lineage extender for tracking feature group data flow.

Wraps calculate_feature to record which feature groups execute, what
features they produce, and the chain dependencies between them.
Provides ASCII and Mermaid visualizations.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from mloda.core.abstract_plugins.function_extender import Extender, ExtenderHook


@dataclass
class LineageNode:
    """One feature group invocation in the lineage graph."""

    feature_group: str
    module: str | None = None
    feature_names: list[str] = field(default_factory=list)
    execution_order: int = 0


class LineageExtender(Extender):
    """Tracks data lineage through feature group execution.

    Uses class-level storage to survive pickle/unpickle during
    mloda's multiprocessing execution.
    """

    _global_nodes: ClassVar[dict[str, list[LineageNode]]] = {}
    _global_counter: ClassVar[dict[str, int]] = {}

    def __init__(self, lineage_id: str | None = None) -> None:
        self._lineage_id = lineage_id or str(uuid.uuid4())
        if self._lineage_id not in LineageExtender._global_nodes:
            LineageExtender._global_nodes[self._lineage_id] = []
            LineageExtender._global_counter[self._lineage_id] = 0

    @property
    def lineage_id(self) -> str:
        return self._lineage_id

    @property
    def _nodes(self) -> list[LineageNode]:
        return LineageExtender._global_nodes[self._lineage_id]

    def wraps(self) -> set[ExtenderHook]:
        return {ExtenderHook.FEATURE_GROUP_CALCULATE_FEATURE}

    def __call__(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        fg_name = _extract_fg_name(func)
        fg_module = _extract_fg_module(func)
        feature_names = _extract_feature_names(args)

        LineageExtender._global_counter[self._lineage_id] += 1
        order = LineageExtender._global_counter[self._lineage_id]

        self._nodes.append(
            LineageNode(
                feature_group=fg_name,
                module=fg_module,
                feature_names=feature_names,
                execution_order=order,
            )
        )

        return func(*args, **kwargs)

    def visualize(self) -> str:
        """ASCII visualisation of the lineage graph."""
        lines = ["Lineage Graph:", ""]
        for node in sorted(self._nodes, key=lambda n: n.execution_order):
            feat_str = ", ".join(node.feature_names) if node.feature_names else "(no features)"
            lines.append(f"  [{node.execution_order}] {node.feature_group}")
            lines.append(f"       features: {feat_str}")

        edges = self.get_edges()
        if edges:
            lines.append("")
            lines.append("  Edges:")
            for src, tgt, feat in edges:
                lines.append(f"    {src} -> {tgt}  ({feat})")
        return "\n".join(lines)

    def visualize_mermaid(self) -> str:
        """Mermaid flowchart of the lineage graph."""
        lines = ["graph TD"]
        node_ids: dict[str, str] = {}
        counter = 0

        for node in sorted(self._nodes, key=lambda n: n.execution_order):
            for feat in node.feature_names:
                nid = f"N{counter}"
                node_ids[feat] = nid
                parts = feat.split("__")
                short = "__".join(parts[-2:]) if len(parts) >= 3 else (parts[-1] if len(parts) >= 2 else feat)
                label = f"{node.feature_group}<br/><i>{short}</i>"
                lines.append(f'    {nid}["{label}"]')
                counter += 1

        for node in sorted(self._nodes, key=lambda n: n.execution_order):
            for feat in node.feature_names:
                parent = _parent_feature(feat)
                if parent and parent in node_ids and feat in node_ids:
                    lines.append(f"    {node_ids[parent]} --> {node_ids[feat]}")

        return "\n".join(lines)

    def reset(self) -> None:
        LineageExtender._global_nodes[self._lineage_id] = []
        LineageExtender._global_counter[self._lineage_id] = 0

    @classmethod
    def clear_all(cls) -> None:
        cls._global_nodes.clear()
        cls._global_counter.clear()

    def _feature_to_fg(self) -> dict[str, str]:
        """Map each produced feature name to its feature group."""
        mapping: dict[str, str] = {}
        for node in self._nodes:
            for feat in node.feature_names:
                mapping[feat] = node.feature_group
        return mapping

    def get_edges(self) -> list[tuple[str, str, str]]:
        """Return (source_fg, target_fg, feature_name) edges."""
        feat_to_fg = self._feature_to_fg()
        edges: list[tuple[str, str, str]] = []
        for node in self._nodes:
            for feat in node.feature_names:
                parent = _parent_feature(feat)
                if parent and parent in feat_to_fg:
                    edges.append((feat_to_fg[parent], node.feature_group, feat))
        return edges


def _extract_fg_name(func: Any) -> str:
    if hasattr(func, "__self__"):
        obj = func.__self__
        return obj.__name__ if isinstance(obj, type) else obj.__class__.__name__
    if hasattr(func, "__qualname__"):
        parts: list[str] = func.__qualname__.split(".")
        if len(parts) >= 2:
            return parts[-2]
    return "unknown"


def _extract_fg_module(func: Any) -> str | None:
    if hasattr(func, "__self__"):
        obj = func.__self__
        return obj.__module__ if isinstance(obj, type) else obj.__class__.__module__
    return None


def _extract_feature_names(args: Any) -> list[str]:
    if len(args) >= 2:
        features = args[1]
        if hasattr(features, "features"):
            return [f.name for f in features.features]
    return []


def _parent_feature(feature_name: str) -> str | None:
    """Derive the parent feature from the chain convention (split on __)."""
    parts = feature_name.rsplit("__", 1)
    if len(parts) == 2 and parts[0]:
        return parts[0]
    return None
