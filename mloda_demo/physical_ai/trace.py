"""Trace view: one line per step in data-flow order, read from the run's OpenTelemetry spans."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import PurePath
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan

from mloda_demo.physical_ai.definition import TO_METRES
from mloda_demo.physical_ai.runner import FOCUS_OBJECT, Run, approaching_objects, nearest_points

ROLES = {
    "DepthLog": "reader",
    "DepthToMetres": "convert",
    "Calibration": "calibrate",
    "NearestDistance": "distance",
    "BrakeRule": "brake",
    "Approaching": "approach",
}
DECLARED = "mloda.declared."


@dataclass(frozen=True)
class TraceLine:
    role: str
    step: str
    version: str
    context: str
    detail: str
    marks: tuple[str, ...] = ()


def _attributes(span: ReadableSpan) -> dict[str, Any]:
    return dict(span.attributes or {})


def _declared(span: ReadableSpan | None) -> dict[str, Any]:
    if span is None:
        return {}
    return {key[len(DECLARED) :]: value for key, value in _attributes(span).items() if key.startswith(DECLARED)}


def _version(span: ReadableSpan) -> str:
    return str(_attributes(span).get("mloda.feature_group.version", "?")).rsplit("-", 1)[-1][:4]


def _step_name(span: ReadableSpan) -> str:
    return str(_attributes(span).get("mloda.feature_group.name", "?")).rsplit(".", 1)[-1]


def trace_lines(result: Run, object_id: int = FOCUS_OBJECT) -> list[TraceLine]:
    """Calculate spans in start order; the reader line folds in its load span. Missing attributes show as '?'."""
    if not result.passed:
        return []
    calculate = sorted((s for s in result.spans if s.name == "mloda.calculate"), key=lambda s: s.start_time or 0)
    load = next((s for s in result.spans if s.name == "mloda.load"), None)
    reader = _declared(load)
    focus = nearest_points(result).loc[object_id]
    raw = focus["depth_raw"]
    encoding = str(reader.get("encoding", result.table["depth_raw"].dtype))
    unit = reader.get("unit", "?")
    assumed = "?"
    lines: list[TraceLine] = []
    for span in calculate:
        step = _step_name(span)
        role = ROLES.get(step)
        if role is None or any(line.role == role for line in lines):
            continue
        declared = _declared(span)
        version = _version(span)
        if role == "reader":
            source = PurePath(str(_attributes(load).get("mloda.data_access.identity", "?"))).name if load else "?"
            name = str(_attributes(load).get("mloda.data_access.format", step)) if load else step
            detail = f"raw {raw:g} ({result.table['depth_raw'].dtype}, unit {unit})"
            lines.append(TraceLine(role, name, str(reader.get("version", version)), f"src {source}", detail))
        elif role == "convert":
            assumed = str(declared.get(f"assumes.{encoding}", "?"))
            factor = TO_METRES.get(assumed)
            product = f"{raw:g} x {factor:g}" if factor is not None else f"{raw:g}"
            lines.append(
                TraceLine(
                    role, step, version, "in depth_raw", f"assumes {assumed}: {product} = {focus['depth_m']:.3g} m"
                )
            )
        elif role == "calibrate":
            frames = f"frame {declared.get('source_frame', '?')} -> {declared.get('frame', '?')}"
            lines.append(TraceLine(role, step, version, f"calib {declared.get('calibration', '?')}", frames))
        elif role == "distance":
            detail = f"object {object_id}: {focus['nearest_distance_m']:.3g} m"
            lines.append(TraceLine(role, step, version, f"unit {declared.get('unit', '?')}", detail))
        elif role == "brake":
            detail = f"object {object_id}: {'brake' if focus['brake'] else 'no brake'}"
            threshold = declared.get("threshold_m", "?")
            threshold = f"{threshold:g}" if isinstance(threshold, float) else threshold
            lines.append(TraceLine(role, step, version, f"threshold {threshold} m", detail))
        else:
            detail = f"approaching: {', '.join(map(str, approaching_objects(result))) or 'none'}"
            lines.append(TraceLine(role, step, version, f"min {declared.get('min_speed_mps', '?')} m/s", detail))
    if "?" not in (unit, assumed) and unit != assumed:
        lines = [_mark(line, unit, assumed) for line in lines]
    return lines


def _mark(line: TraceLine, unit: str, assumed: str) -> TraceLine:
    marks = {"reader": (f"unit {unit}",), "convert": (f"assumes {assumed}",)}.get(line.role, ())
    return TraceLine(line.role, line.step, line.version, line.context, line.detail, marks)


def header(result: Run) -> str:
    run_id = next((str(_attributes(s)["mloda.run.id"]) for s in result.spans if "mloda.run.id" in _attributes(s)), "?")
    return f"run {run_id[-4:]} · {result.reader.sensor} · {result.label}"


def format_trace(result: Run, object_id: int = FOCUS_OBJECT) -> str:
    rows = [
        (line.role, line.step, f"v {line.version}", line.context, line.detail)
        for line in trace_lines(result, object_id)
    ]
    widths = [max((len(row[i]) for row in rows), default=0) for i in range(4)]
    body = ["   ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:4])) + "   " + row[4] for row in rows]
    return "\n".join([header(result), *body])


def trace_html(result: Run, object_id: int = FOCUS_OBJECT) -> str:
    """The plain-text trace as <pre>, with the unit mismatch marked."""
    text = html.escape(format_trace(result, object_id))
    for line in trace_lines(result, object_id):
        for mark in line.marks:
            text = text.replace(html.escape(mark), f"<mark>{html.escape(mark)}</mark>", 1)
    return f'<pre class="trace">{text}</pre>'
