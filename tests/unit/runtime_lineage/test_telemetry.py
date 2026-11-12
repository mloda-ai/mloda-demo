import pytest
from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.community.extenders.otel.otel_extender import OtelExtender
from mloda.community.extenders.otel.otel_metrics_extender import OtelMetricsExtender
from openlineage.client.transport.http import HttpTransport

from mloda_demo.runtime_lineage.telemetry import LINEAGE_URL, OTLP_ENDPOINT, telemetry


def test_without_a_stack_everything_stays_in_memory() -> None:
    recorded = telemetry({})
    try:
        assert {type(extender) for extender in recorded.extenders} == {OtelExtender, OpenLineageExtender}
        assert recorded.meter_provider is None and recorded.transport.forward is None
        assert recorded.latest_run() is None
    finally:
        recorded.close()


def test_with_the_stack_configured_the_hooks_also_export() -> None:
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    # Nothing listens on the discard port; building the exporters sends nothing.
    recorded = telemetry({OTLP_ENDPOINT: "http://127.0.0.1:9/", LINEAGE_URL: "http://127.0.0.1:9"})
    try:
        kinds = {type(extender) for extender in recorded.extenders}
        assert kinds == {OtelExtender, OtelMetricsExtender, OpenLineageExtender}
        assert recorded.meter_provider is not None
        assert isinstance(recorded.transport.forward, HttpTransport)
    finally:
        recorded.close()
