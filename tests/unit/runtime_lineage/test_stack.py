import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")  # arrives with the OpenLineage client; not a dependency of its own

STACK = Path(__file__).resolve().parents[3] / "stack" / "runtime_lineage"
CONFIGS = ("compose.yaml", "otel-collector.yaml", "tempo.yaml", "prometheus.yaml")


def test_the_stack_configs_parse() -> None:
    for name in CONFIGS:
        assert isinstance(yaml.safe_load((STACK / name).read_text(encoding="utf-8")), dict), name


def test_every_image_is_pinned() -> None:
    compose = yaml.safe_load((STACK / "compose.yaml").read_text(encoding="utf-8"))
    images = [service["image"] for service in compose["services"].values()]
    assert images
    for image in images:
        tag = image.rsplit("/", 1)[-1].partition(":")[2]
        assert tag and tag != "latest", image


def test_the_dashboard_uses_only_provisioned_data_sources() -> None:
    provisioned = yaml.safe_load(
        (STACK / "grafana" / "provisioning" / "datasources" / "datasources.yaml").read_text(encoding="utf-8")
    )
    known = {source["uid"] for source in provisioned["datasources"]}
    dashboard = json.loads((STACK / "grafana" / "dashboards" / "mloda.json").read_text(encoding="utf-8"))
    used = {panel["datasource"]["uid"] for panel in dashboard["panels"] if "datasource" in panel}
    used |= {target["datasource"]["uid"] for panel in dashboard["panels"] for target in panel.get("targets", [])}
    assert used and used <= known
