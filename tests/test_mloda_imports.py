"""Tests to verify mloda dependencies can be imported."""


def test_mloda_provider_imports() -> None:
    """Verify mloda.provider module imports work."""
    from mloda.provider import ComputeFramework, FeatureGroup

    assert FeatureGroup is not None
    assert ComputeFramework is not None


def test_mloda_core_imports() -> None:
    """Verify mloda.core module imports work."""
    from mloda.core.abstract_plugins.function_extender import Extender

    assert Extender is not None


def test_otel_extender_imports() -> None:
    """Verify the registry OtelExtender and the SDK in-memory exporter import."""
    from mloda.community.extenders.otel import OtelExtender
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    assert OtelExtender is not None
    assert InMemorySpanExporter is not None


def test_mloda_testing_imports() -> None:
    """Verify mloda.testing module imports work."""
    from mloda.testing.base import FeatureGroupTestBase

    assert FeatureGroupTestBase is not None
