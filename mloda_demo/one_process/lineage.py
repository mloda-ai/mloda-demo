"""One mloda call, the OpenLineage events it emits, and the pipeline picture drawn from them."""

from __future__ import annotations

import html
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from io import StringIO
from typing import Any

import matplotlib
import pandas as pd
from mloda.community.extenders.openlineage import OpenLineageExtender
from mloda.provider import FeatureGroup
from mloda.user import Feature, Options, PluginCollector, mloda
from openlineage.client.client import OpenLineageClient
from openlineage.client.event_v2 import RunEvent, RunState
from openlineage.client.transport.transport import Config, Transport

from mloda_demo.one_process.chain import root_of, source_of
from mloda_demo.physical_ai.plot import pipeline_graph

SOURCE = "source"
COLUMN_SHARE = 24  # percent of the page width per pipeline column


class RecordingTransport(Transport):
    """Keeps every emitted event in memory instead of sending it to a lineage server."""

    kind = "recording"
    config_class = Config

    def __init__(self, config: Config | None = None) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: Any) -> None:
        if isinstance(event, RunEvent):
            self.events.append(event)

    def close(self, timeout: float = -1) -> bool:
        return True


@dataclass(frozen=True)
class Step:
    """One completed feature group run as OpenLineage recorded it: what it read and what it wrote."""

    job: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Lineage:
    """The result of one call: a table per requested feature and the completed steps in run order."""

    features: tuple[str, ...]
    tables: dict[str, pd.DataFrame]
    steps: tuple[Step, ...]

    def chain(self, feature: str) -> list[str]:
        """Step labels from the source to the requested feature, read off the events."""
        by_output = {name: step for step in self.steps for name in step.outputs}
        labels: list[str] = []
        name = feature
        while name in by_output:
            step = by_output[name]
            labels.append(label(step.job))
            # The chain continues at the name's own source when the step read one, else at what it read.
            source = source_of(name) if "__" in name else None
            earlier = [source] if source in by_output else [read for read in step.inputs if read in by_output]
            if not earlier:
                break
            producers = {by_output[read].job for read in earlier}
            if len(producers) != 1:
                raise ValueError(f"{name}: read from several steps {sorted(producers)}; the picture draws chains")
            name = earlier[0]
        return list(reversed(labels))

    def chains(self) -> list[list[str]]:
        return [self.chain(feature) for feature in self.features]

    def svg(self) -> str:
        chains = self.chains()
        figure = pipeline_graph(chains)
        buffer = StringIO()
        # A fresh salt per picture keeps element ids unique when several pictures share one page.
        with matplotlib.rc_context({"svg.hashsalt": uuid.uuid4().hex}):
            figure.savefig(buffer, format="svg", transparent=True, bbox_inches="tight", metadata={"Date": None})
        # Every column takes the same share of the page, so boxes keep one size from picture to picture.
        share = min(100, COLUMN_SHARE * max(len(chain) for chain in chains))
        return re.sub(r'width="[\d.]+pt" height="[\d.]+pt"', f'width="{share}%"', buffer.getvalue(), count=1)

    def html(self) -> str:
        caption = " · ".join(f"{html.escape(name)} {rows(len(self.tables[name]))}" for name in self.features)
        return f'<figure class="lineage">{self.svg()}<figcaption>{caption}</figcaption></figure>'

    def _mime_(self) -> tuple[str, str]:
        """marimo shows the picture when the call is a cell's last expression."""
        return "text/html", self.html()


def label(job: str) -> str:
    """A step's class name as it reads in a feature name: MonthlyPayment -> monthly_payment."""
    name = job.rsplit(".", 1)[-1]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def rows(count: int) -> str:
    return "1 row" if count == 1 else f"{count} rows"


def run(features: Iterable[str], feature_groups: Iterable[type[FeatureGroup]]) -> Lineage:
    """One mloda call over the requested chains.

    Each chain runs in its own group, keyed by the source it starts with, so mloda keeps sources of
    different length apart instead of joining them.
    """
    names = tuple(features)
    transport = RecordingTransport()
    client = OpenLineageClient(transport=transport)
    if client.transport is not transport:
        raise RuntimeError("OPENLINEAGE_DISABLED is set; the picture needs the events")
    frames = mloda.run_all(
        [Feature(name, Options(group={SOURCE: root_of(name)})) for name in names],
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups(set(feature_groups)),
        function_extender={OpenLineageExtender(client=client)},
    )
    tables = {name: frame for frame in frames for name in names if name in frame.columns}
    missing = [name for name in names if name not in tables]
    if missing:
        raise ValueError(f"no result table for {missing}")
    steps = tuple(
        Step(event.job.name, tuple(d.name for d in event.inputs or ()), tuple(d.name for d in event.outputs or ()))
        for event in transport.events
        if event.eventType == RunState.COMPLETE
    )
    return Lineage(names, tables, steps)
