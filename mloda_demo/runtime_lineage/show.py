"""`show` formats what a run returned and what its hooks recorded. Nothing is computed here."""

from __future__ import annotations

import html
import uuid
from collections.abc import Sequence
from io import StringIO

import matplotlib
from mloda.user import RunResult
from openlineage.client.event_v2 import RunEvent, RunState
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from mloda_demo.physical_ai.plot import pipeline_graph
from mloda_demo.retail.runs import Shown, table
from mloda_demo.runtime_lineage.telemetry import Telemetry, parent_of, trace_of

EVIDENCE = {
    "mloda.data_access.identity": "file",
    "mloda.declared.policy.verdict": "rule",
    "mloda.declared.chart": "chart",
    "error.type": "refused",
}
GROUP, OPERATION = "mloda.feature_group.name", "mloda.operation.name"


def show(answer: RunResult | BaseException, telemetry: Telemetry, evidence: bool = True) -> Shown:
    """The number with its cited rows, or the refusal; then the run's trace, one line per step, and its lineage.

    With `evidence=False`, only the number and how the run ended, as a dashboard would show it.
    """
    if isinstance(answer, BaseException):
        head, run_id = refusal(answer), telemetry.refused_run(answer)
    else:
        head, run_id = value(answer), answer.run_id
    if run_id is None:
        return Shown(head)
    spans, events = telemetry.spans(run_id), telemetry.events(run_id)
    rows = steps(spans)
    if not evidence:
        return Shown(head + outcome(rows))
    trace = table(rows) if rows else outcome(rows)
    return Shown(head + ids(run_id, spans, events) + trace + picture(events))


def outcome(rows: Sequence[dict[str, str]]) -> str:
    failed = sum(1 for row in rows if row["status"] == "error")
    text = f"{len(rows)} steps, {f'{failed} failed' if failed else 'all ok'}" if rows else "no step ran"
    return f'<p class="retail-label">{text}</p>'


def value(result: RunResult) -> str:
    frame = result[0]
    column = next(name for name in frame.column_names if name.endswith("~value"))
    name = column.removesuffix("~value")
    shown = f'<p class="retail-value">{html.escape(name)}, {frame.column(column)[0].as_py():,.2f}</p>'
    if f"{name}~origins" not in frame.column_names:
        return shown
    cited = "\n".join(frame.column(f"{name}~origins")[0].as_py())
    return f'{shown}<p class="retail-label">Summed from</p><pre><code>{html.escape(cited)}</code></pre>'


def refusal(error: BaseException) -> str:
    return f'<pre class="retail-refusal"><code>{html.escape(f"{type(error).__name__}: {error}")}</code></pre>'


def short(name: object) -> str:
    return str(name).rsplit(".", 1)[-1]


def ids(run_id: str, spans: Sequence[ReadableSpan], events: Sequence[RunEvent]) -> str:
    """The one ID: the trace, the lineage run, and how many of the run's lineage steps point at the trace."""
    trace_id = format(spans[0].context.trace_id, "032x") if spans else "?"
    step_events = [event for event in events if event.eventType != RunState.START and parent_of(event) == run_id]
    linked = sum(1 for event in step_events if trace_of(event) == trace_id)
    text = f"trace {trace_id} · lineage run {run_id} · {linked} of {len(step_events)} lineage steps point at this trace"
    return f'<p class="retail-label">{html.escape(text)}</p>'


def steps(spans: Sequence[ReadableSpan]) -> list[dict[str, str]]:
    """One row per calculate step; a step's evidence is read from all its spans, so the reader's row names its file."""
    rows = []
    for span in spans:
        attributes = dict(span.attributes or {})
        if attributes.get(OPERATION) != "calculate":
            continue
        evidence: dict[str, str] = {}
        for other in spans:
            if (other.attributes or {}).get(GROUP) == attributes.get(GROUP):
                found = {k: v for k, v in (other.attributes or {}).items() if k in EVIDENCE}
                # An error type is a dotted class path; the other values are shown as recorded.
                evidence |= {EVIDENCE[k]: short(v) if k == "error.type" else str(v) for k, v in found.items()}
        rows.append(
            {
                "step": short(attributes.get(GROUP, "?")),
                "status": "error" if span.status.status_code is StatusCode.ERROR else "ok",
                "evidence": ", ".join(f"{word} {found}" for word, found in evidence.items()),
            }
        )
    return rows


def chain(events: Sequence[RunEvent]) -> list[str]:
    """The file the first step read, then each step in run order; a step that failed says so."""
    ended = [e for e in events if e.eventType in (RunState.COMPLETE, RunState.FAIL) and parent_of(e) is not None]
    source = [d.name for e in ended[:1] for d in (e.inputs or ())[:1]]
    return source + [short(e.job.name) + (" (failed)" if e.eventType == RunState.FAIL else "") for e in ended]


def picture(events: Sequence[RunEvent]) -> str:
    boxes = chain(events)
    if not boxes:
        return ""
    buffer = StringIO()
    # A fresh salt keeps element ids unique when several pictures share one page.
    with matplotlib.rc_context({"svg.hashsalt": uuid.uuid4().hex}):
        pipeline_graph([boxes]).savefig(
            buffer, format="svg", transparent=True, bbox_inches="tight", metadata={"Date": None}
        )
    return f'<figure class="retail-plan">{buffer.getvalue()}</figure>'
