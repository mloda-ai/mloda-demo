"""Run a definition over the clip and keep its OpenTelemetry trace."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mloda.community.extenders.otel import OtelExtender
from mloda.provider import FeatureGroup, FeatureResolutionError
from mloda.user import Feature, Options, PluginCollector, mloda
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mloda_demo.physical_ai.clip import CLIP_DIR, nearest_in_corridor
from mloda_demo.physical_ai.definition import DepthToMetres, NearestAhead, StopRule
from mloda_demo.physical_ai.readers import REQUIRES_DECLARED_SCALE, DepthFrameReader, DepthFrames

FEATURE_GROUPS: frozenset[type[FeatureGroup]] = frozenset({DepthFrames, DepthToMetres, NearestAhead, StopRule})
FEATURES = ("frame", "t_s", "source", "depth_raw", "depth_scale", "depth_m", "nearest_ahead_m", "stop")


@dataclass(frozen=True)
class Run:
    label: str
    reader: type[DepthFrameReader] | None
    table: pd.DataFrame
    spans: tuple[ReadableSpan, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None


def run(
    reader: type[DepthFrameReader] | None,
    *,
    label: str | None = None,
    check: bool = False,
    features: Iterable[str] = FEATURES,
    feature_groups: Iterable[type[FeatureGroup]] = FEATURE_GROUPS,
) -> Run:
    """One mloda run with an in-memory span exporter; a resolution failure becomes Run.error."""
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    group: dict[Any, Any] = {}
    if reader is not None:
        group[reader] = str(CLIP_DIR)
    if check:
        # The conversion's requirement travels with the request, so an undeclared reader is refused at plan time.
        group[REQUIRES_DECLARED_SCALE] = DepthToMetres.get_class_name()
    name = label or (reader.label if reader is not None else "welded")
    try:
        frames = mloda.run_all(
            [Feature(feature, Options(group=dict(group))) for feature in features],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(set(feature_groups)),
            function_extender={OtelExtender(tracer_provider=provider)},
        )
    except FeatureResolutionError as error:
        return Run(name, reader, pd.DataFrame(), tuple(exporter.get_finished_spans()), rejection(error))
    # A linear pandas chain keeps row order, so the per-step frames line up.
    if any(not frame.index.equals(frames[0].index) for frame in frames):
        raise ValueError("per-step result frames are not row-aligned")
    return Run(name, reader, pd.concat(list(frames), axis=1), tuple(exporter.get_finished_spans()))


def rejection(error: FeatureResolutionError) -> str:
    """The reason a candidate was eliminated, without resolver boilerplate."""
    reasons = [elimination.reason for elimination in error.result.eliminations.values()]
    return "; ".join(reasons) if reasons else str(error)


def closest_frame(result: Run) -> int:
    """The frame with the nearest reading: the chair. The last frame when no reading is valid."""
    nearest = result.table["nearest_ahead_m"]
    if nearest.isna().all():
        return int(result.table["frame"].iloc[-1])
    return int(result.table.loc[nearest.idxmin(), "frame"])


def at_frame(result: Run, frame: int) -> pd.Series:
    return result.table[result.table["frame"] == frame].iloc[0]


def raw_nearest(result: Run, frame: int) -> float:
    """The raw reading behind the nearest metres, for the trace."""
    return nearest_in_corridor(at_frame(result, frame)["depth_raw"])
