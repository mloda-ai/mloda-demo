"""Run the shared definition through one reader and keep its OpenTelemetry trace."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mloda.community.extenders.otel import OtelExtender
from mloda.provider import FeatureResolutionError
from mloda.user import Feature, Options, PluginCollector, mloda
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mloda_demo.physical_ai.definition import Approaching, BrakeRule, Calibration, DepthToMetres, NearestDistance
from mloda_demo.physical_ai.readers import DepthLog, DepthLogReader, DepthReaderA

FEATURE_GROUPS = frozenset({DepthLog, DepthToMetres, Calibration, NearestDistance, BrakeRule, Approaching})
FEATURES = ("frame", "timestamp", "object_id", "depth_raw", "depth_m", "x_m", "y_m", "nearest_distance_m", "brake")
FOCUS_OBJECT = 7


@dataclass(frozen=True)
class Run:
    label: str
    reader: type[DepthLogReader]
    table: pd.DataFrame
    spans: tuple[ReadableSpan, ...]
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None


def run(
    reader: type[DepthLogReader],
    *,
    label: str = "offline",
    fault: bool = False,
    check_units: bool = False,
    replay_until: int | None = None,
    features: Iterable[str] = FEATURES,
) -> Run:
    """One mloda run with an in-memory span exporter; a resolution failure becomes Run.error."""
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    group: dict[Any, Any] = {reader: reader.log_path(), "fault": fault}
    if replay_until is not None:
        group["replay_until"] = replay_until
    if check_units:
        # The conversion's declared units travel with the request, so the reader refuses a mismatch at plan time.
        group["assumed_units"] = tuple(sorted(DepthToMetres.assumed_units(fault).items()))
    try:
        frames = mloda.run_all(
            [Feature(name, Options(group=dict(group))) for name in features],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(set(FEATURE_GROUPS)),
            function_extender={OtelExtender(tracer_provider=provider)},
        )
    except FeatureResolutionError as error:
        return Run(label, reader, pd.DataFrame(), tuple(exporter.get_finished_spans()), rejection(error))
    # A linear pandas chain keeps row order, so the per-step frames line up.
    table = pd.concat([frame.reset_index(drop=True) for frame in frames], axis=1)
    return Run(label, reader, table, tuple(exporter.get_finished_spans()))


def compare(
    reader: type[DepthLogReader], *, fault: bool = False, frame: int | None = None, features: Iterable[str] = FEATURES
) -> dict[str, Run]:
    """The n-table's contexts: Device A logs offline, Device A replayed online up to a frame, and the given reader."""
    features = tuple(features)
    return {
        "offline": run(DepthReaderA, features=features),
        "online": run(DepthReaderA, label="online", replay_until=frame, features=features),
        reader.label: run(reader, label=reader.label, fault=fault, features=features),
    }


def rejection(error: FeatureResolutionError) -> str:
    """The reason a candidate was eliminated, without resolver boilerplate."""
    reasons = [elimination.reason for elimination in error.result.eliminations.values()]
    return "; ".join(reasons) if reasons else str(error)


def nearest_points(result: Run, frame: int | None = None) -> pd.DataFrame:
    """Each object's nearest sample at one frame (default: the last one)."""
    table = result.table
    frame = int(table["frame"].max()) if frame is None else frame
    rows = table[table["frame"] == frame]
    distance = (rows["x_m"] ** 2 + rows["y_m"] ** 2) ** 0.5
    return rows.loc[distance.groupby(rows["object_id"]).idxmin()].set_index("object_id").sort_index()


def n_table(runs: Mapping[str, Run], object_id: int = FOCUS_OBJECT) -> pd.DataFrame:
    """Focus object distance and brake decision, one column per run; a brake answer differing from the first is upper case."""
    asked = all(result.passed and "approaching" in result.table for result in runs.values())
    index = [f"object {object_id}", "brake", *(["approaching"] if asked else [])]
    columns: dict[str, list[str]] = {}
    first_brake: bool | None = None
    for name, result in runs.items():
        if not result.passed:
            columns[name] = ["error"] * len(index)
            continue
        point = nearest_points(result).loc[object_id]
        brake = bool(point["brake"])
        first_brake = brake if first_brake is None else first_brake
        answer = "yes" if brake else "no"
        column = [f"{point['nearest_distance_m']:.3g} m", answer if brake == first_brake else answer.upper()]
        if asked:
            column.append(", ".join(map(str, approaching_objects(result))) or "none")
        columns[name] = column
    return pd.DataFrame(columns, index=index)


def approaching_objects(result: Run) -> list[int]:
    """Objects approaching at the last frame, when the run requested `approaching`."""
    points = nearest_points(result)
    return [int(object_id) for object_id in points.index[points["approaching"].astype(bool)]]


def all_passed(runs: Iterable[Run]) -> bool:
    return all(result.passed for result in runs)
