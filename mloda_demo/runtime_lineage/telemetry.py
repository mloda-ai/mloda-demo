"""The spans and lineage events of the demo's runs: kept in memory, and sent to the local stack when it is configured."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.community.extenders.otel.otel_extender import OtelExtender
from mloda.community.extenders.otel.otel_metrics_extender import OtelMetricsExtender
from mloda.steward import Extender
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import RunEvent
from openlineage.client.transport.http import HttpConfig, HttpTransport
from openlineage.client.transport.transport import Transport
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import AggregationTemporality, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mloda_demo.one_process.lineage import RecordingTransport

SERVICE = "tax-office-assistant"
OTLP_ENDPOINT = "OTEL_EXPORTER_OTLP_ENDPOINT"  # the Collector, e.g. http://localhost:4318
LINEAGE_URL = "OPENLINEAGE_URL"  # Marquez, e.g. http://localhost:5002
EXPORT_TIMEOUT = 1.0  # seconds
RUN_ID = "mloda.run.id"
ROOT_SPAN = "mloda.run"
logger = logging.getLogger(__name__)


class ForwardingTransport(RecordingTransport):
    """Keeps every event and passes it on to a lineage server when one is given."""

    def __init__(self, forward: Transport | None = None) -> None:
        super().__init__()
        self.forward = forward

    def emit(self, event: Any) -> None:
        super().emit(event)
        if self.forward is None:
            return
        try:
            self.forward.emit(event)
        except Exception as error:  # noqa: BLE001  whatever the transport raises
            # The record above is what the demo shows; a server that fails once is left alone.
            logger.warning("lineage server not reachable (%s); events stay in memory", type(error).__name__)
            self.forward = None


@dataclass(frozen=True)
class Telemetry:
    """The hooks passed to a run, and what they recorded."""

    extenders: frozenset[Extender]
    tracer_provider: TracerProvider
    meter_provider: MeterProvider | None
    recorded_spans: InMemorySpanExporter
    transport: ForwardingTransport

    def latest_run(self) -> str | None:
        """The id of the run that ended last."""
        root = self._latest_root()
        return str((root.attributes or {})[RUN_ID]) if root is not None else None

    def refused_run(self, error: BaseException) -> str | None:
        """The run that raised `error`: the last run, if it failed with that type. A refusal carries no run id."""
        root = self._latest_root()
        if root is None or root.status.status_code is not StatusCode.ERROR:
            return None
        failed_with = str((root.attributes or {}).get("error.type", ""))
        return str((root.attributes or {})[RUN_ID]) if failed_with.endswith(type(error).__name__) else None

    def _latest_root(self) -> ReadableSpan | None:
        roots = [span for span in self.recorded_spans.get_finished_spans() if span.name == ROOT_SPAN]
        return roots[-1] if roots else None

    def spans(self, run_id: str) -> list[ReadableSpan]:
        spans = [s for s in self.recorded_spans.get_finished_spans() if (s.attributes or {}).get(RUN_ID) == run_id]
        return sorted(spans, key=lambda span: span.start_time or 0)

    def events(self, run_id: str) -> list[RunEvent]:
        """The root run's events and those of its steps, in the order they were emitted."""
        return [event for event in self.transport.events if run_id in (event.run.runId, parent_of(event))]

    def close(self) -> None:
        self.tracer_provider.shutdown()
        if self.meter_provider is not None:
            self.meter_provider.shutdown()


def parent_of(event: RunEvent) -> str | None:
    parent = (event.run.facets or {}).get("parent")
    return str(parent.run.runId) if parent is not None else None  # type: ignore[attr-defined]


def trace_of(event: RunEvent) -> str | None:
    """The trace a step's lineage event points at; root events carry none."""
    trace = (event.run.facets or {}).get("mlodaTrace")
    return str(trace.traceId) if trace is not None else None  # type: ignore[attr-defined]


def telemetry(environ: Mapping[str, str] | None = None) -> Telemetry:
    """Fresh providers per call, so nothing is shared between notebooks or tests. No global provider is set."""
    environ = os.environ if environ is None else environ
    resource = Resource.create({"service.name": SERVICE})
    recorded = InMemorySpanExporter()
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(recorded))
    meter_provider = None
    extenders: set[Extender] = set()
    endpoint = environ.get(OTLP_ENDPOINT, "").rstrip("/")
    if endpoint:
        # Short timeouts and no retries: a stack that is down must not stall a run.
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as error:
            raise RuntimeError(f"{OTLP_ENDPOINT} is set: install the stack extra or unset it") from error

        spans = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", timeout=EXPORT_TIMEOUT)
        tracer_provider.add_span_processor(BatchSpanProcessor(spans, schedule_delay_millis=500))
        # Deltas: the Collector turns them back into cumulative series.
        delta: dict[type, AggregationTemporality] = {
            Counter: AggregationTemporality.DELTA,
            Histogram: AggregationTemporality.DELTA,
        }
        exporter = OTLPMetricExporter(
            endpoint=f"{endpoint}/v1/metrics", timeout=EXPORT_TIMEOUT, preferred_temporality=delta
        )
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=2000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
        extenders.add(OtelMetricsExtender(meter_provider=meter_provider))
    url = environ.get(LINEAGE_URL, "")
    forward = HttpTransport(HttpConfig(url=url, timeout=EXPORT_TIMEOUT, retry={"total": 0})) if url else None
    transport = ForwardingTransport(forward)
    client = OpenLineageClient(transport=transport)
    if client.transport is not transport:
        raise RuntimeError("OPENLINEAGE_DISABLED is set; the demo needs the lineage events")
    extenders |= {OtelExtender(tracer_provider=tracer_provider), OpenLineageExtender(client=client)}
    return Telemetry(frozenset(extenders), tracer_provider, meter_provider, recorded, transport)
