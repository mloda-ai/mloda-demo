"""Trace view: one line per step in data-flow order, read from the run's OpenTelemetry spans."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan

from mloda_demo.physical_ai.definition import ASSUMED_SCALE, ASSUMED_UNIT
from mloda_demo.physical_ai.runner import Run, at_frame, closest_frame, raw_nearest

ROLES = {"DepthFrames": "reader", "DepthToMetres": "convert", "NearestAhead": "nearest", "StopRule": "stop"}
DECLARED = "mloda.declared."


@dataclass(frozen=True)
class TraceLine:
    role: str
    step: str
    version: str
    context: str
    detail: str
    marks: tuple[str, ...] = ()


def _attributes(span: ReadableSpan | None) -> dict[str, Any]:
    return dict(span.attributes or {}) if span is not None else {}


def _declared(span: ReadableSpan | None) -> dict[str, Any]:
    return {key[len(DECLARED) :]: value for key, value in _attributes(span).items() if key.startswith(DECLARED)}


def _version(span: ReadableSpan) -> str:
    return str(_attributes(span).get("mloda.feature_group.version", "?")).rsplit("-", 1)[-1][:4]


def _step_name(span: ReadableSpan) -> str:
    return str(_attributes(span).get("mloda.feature_group.name", "?")).rsplit(".", 1)[-1]


def calculate_spans(result: Run) -> list[ReadableSpan]:
    return sorted((s for s in result.spans if s.name == "mloda.calculate"), key=lambda s: s.start_time or 0)


def load_span(result: Run) -> ReadableSpan | None:
    return next((s for s in result.spans if s.name == "mloda.load"), None)


def chain(result: Run) -> list[str]:
    """Feature group names in data-flow order; the root group shows as the reader that loaded for it."""
    load = load_span(result)
    reader = str(_attributes(load).get("mloda.data_access.format", "?")) if load is not None else None
    parent = load.parent.span_id if load is not None and load.parent is not None else None
    names = []
    for span in calculate_spans(result):
        is_root = parent is not None and span.context is not None and span.context.span_id == parent
        names.append(reader if is_root and reader is not None else _step_name(span))
    return names


def trace_lines(result: Run, frame: int | None = None) -> list[TraceLine]:
    """One line per step for the given frame (default: the closest one). Missing attributes show as '?'."""
    if not result.passed:
        return []
    frame = closest_frame(result) if frame is None else frame
    row = at_frame(result, frame)
    raw = raw_nearest(result, frame)
    load = load_span(result)
    reader = _declared(load)
    encoding = str(reader.get("encoding", row["depth_raw"].dtype))
    scale = str(reader.get("scale", "?"))
    undeclared = scale == "undeclared"
    nearest = float(row["nearest_ahead_m"])
    lines: list[TraceLine] = []
    for span in calculate_spans(result):
        step = _step_name(span)
        role = ROLES.get(step)
        if role is None or any(line.role == role for line in lines):
            continue
        declared = _declared(span)
        version = _version(span)
        if role == "reader":
            name = str(_attributes(load).get("mloda.data_access.format", step)) if load is not None else step
            detail = f"raw {raw:g} ({encoding}, scale {scale})"
            lines.append(TraceLine(role, name, str(reader.get("version", version)), f"src {row['source']}", detail))
        elif role == "convert":
            if undeclared:
                unit = str(declared.get(f"assumes.{encoding}", ASSUMED_UNIT.get(encoding, "?")))
                factor = 1 / ASSUMED_SCALE.get(encoding, float("nan"))
                context, detail = "in depth_raw", f"assumes {unit}: {raw:g} x {factor:g} = {nearest:.2f} m"
            else:
                context, detail = (
                    "in depth_raw",
                    f"scale {scale}: {raw:g} / {float(row['depth_scale']):g} = {nearest:.2f} m",
                )
            lines.append(TraceLine(role, step, version, context, detail))
        elif role == "nearest":
            lines.append(
                TraceLine(role, step, version, f"unit {declared.get('unit', '?')}", f"frame {frame}: {nearest:.2f} m")
            )
        else:
            threshold = declared.get("threshold_m", "?")
            threshold = f"{threshold:g}" if isinstance(threshold, float) else str(threshold)
            detail = "stop" if bool(row["stop"]) else "go"
            lines.append(TraceLine(role, step, version, f"threshold {threshold} m", detail))
    if undeclared:
        lines = [_mark(line, encoding) for line in lines]
    return lines


def _mark(line: TraceLine, encoding: str) -> TraceLine:
    marks = {"reader": ("scale undeclared",), "convert": (f"assumes {ASSUMED_UNIT.get(encoding, '?')}",)}.get(
        line.role, ()
    )
    return TraceLine(line.role, line.step, line.version, line.context, line.detail, marks)


def header(result: Run, frame: int | None = None) -> str:
    run_id = next((str(_attributes(s)["mloda.run.id"]) for s in result.spans if "mloda.run.id" in _attributes(s)), "?")
    frame = closest_frame(result) if frame is None else frame
    return f"run {run_id[-4:]} · {result.label} · frame {frame}, t = {float(at_frame(result, frame)['t_s']):.1f} s"


def format_trace(result: Run, frame: int | None = None) -> str:
    rows = [
        (line.role, line.step, f"v {line.version}", line.context, line.detail) for line in trace_lines(result, frame)
    ]
    widths = [max((len(row[i]) for row in rows), default=0) for i in range(4)]
    body = ["   ".join(cell.ljust(widths[i]) for i, cell in enumerate(row[:4])) + "   " + row[4] for row in rows]
    return "\n".join([header(result, frame), *body])


def trace_html(result: Run, frame: int | None = None) -> str:
    """The plain-text trace as <pre>, with the undeclared scale and the assumption marked."""
    text = html.escape(format_trace(result, frame))
    for line in trace_lines(result, frame):
        for mark in line.marks:
            text = text.replace(html.escape(mark), f"<mark>{html.escape(mark)}</mark>", 1)
    return f'<pre class="trace">{text}</pre>'
