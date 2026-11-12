from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from mloda.community.feature_groups.experimental.german_ledger.skr import ChartConflict, SkrAccountFeatureGroup
from mloda.user import Feature, PluginCollector, RunResult, mloda
from openlineage.client.event_v2 import RunEvent, RunState
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import StatusCode

from mloda_demo.runtime_lineage import (
    BOOKS,
    CLIENT_A,
    CLIENT_B,
    CLIENT_B_CORRECTED,
    CLIENT_C,
    READS_SKR03,
    READS_SKR04,
    show,
    telemetry,
)
from mloda_demo.runtime_lineage.office import Skr03Accounts
from mloda_demo.runtime_lineage.show import chain
from mloda_demo.runtime_lineage.telemetry import Telemetry, parent_of, trace_of

BATCH = "EXTF_Buchungsstapel.csv"


@pytest.fixture
def recorded() -> Iterator[Telemetry]:
    hooks = telemetry({})  # an empty environment: nothing leaves the process
    yield hooks
    hooks.close()


def ask(recorded: Telemetry, name: str, reader: str, folder: Path, office: PluginCollector = READS_SKR04) -> RunResult:
    return mloda.run_all(
        [Feature(name, options={reader: str(folder)})],
        compute_frameworks=["PyArrowTable"],
        data_access_collection=BOOKS,
        plugin_collector=office,
        function_extender=set(recorded.extenders),
    )


def total(result: RunResult, concept: str) -> tuple[Decimal, list[str]]:
    frame, name = result[0], f"{concept}__sources"
    return frame.column(f"{name}~value")[0].as_py(), frame.column(f"{name}~origins")[0].as_py()


def calculate_spans(recorded: Telemetry, run_id: str) -> dict[str, ReadableSpan]:
    """The run's calculate spans by the short name of their feature group."""
    return {
        str(span.attributes["mloda.feature_group.name"]).rsplit(".", 1)[-1]: span
        for span in recorded.spans(run_id)
        if span.attributes.get("mloda.operation.name") == "calculate"
    }


def step_events(recorded: Telemetry, run_id: str) -> list[RunEvent]:
    return [e for e in recorded.events(run_id) if parent_of(e) == run_id and e.eventType != RunState.START]


def identity(recorded: Telemetry, run_id: str) -> str:
    return next(
        str(span.attributes["mloda.data_access.identity"])
        for span in recorded.spans(run_id)
        if "mloda.data_access.identity" in span.attributes
    )


def test_one_answer_has_one_trace_and_one_lineage_run(recorded: Telemetry) -> None:
    result = ask(recorded, "revenue__sources", "DatevExtfReader", CLIENT_A)
    value, cited = total(result, "revenue")
    assert value == Decimal("22485.06")
    assert len(cited) == 3 and all(origin.startswith(f"{BATCH}@") for origin in cited)

    spans = recorded.spans(result.run_id)
    assert len({span.context.trace_id for span in spans}) == 1
    assert recorded.latest_run() == result.run_id
    by_step_run = {
        span.attributes.get("mloda.step.run_id"): span for span in calculate_spans(recorded, result.run_id).values()
    }
    events = step_events(recorded, result.run_id)
    assert [e.eventType for e in events] == [RunState.COMPLETE] * 4
    for event in events:
        # Each lineage step points at its own calculate span.
        span = by_step_run[event.run.runId]
        assert trace_of(event) == format(span.context.trace_id, "032x")
        assert event.run.facets["mlodaTrace"].spanId == format(span.context.span_id, "016x")
    roots = [e for e in recorded.events(result.run_id) if parent_of(e) is None]
    assert {e.run.runId for e in roots} == {result.run_id}


def test_another_export_takes_another_path_to_the_same_total(recorded: Telemetry) -> None:
    batch = ask(recorded, "revenue__sources", "DatevExtfReader", CLIENT_A)
    dossier = ask(recorded, "revenue__sources", "GdpduReader", CLIENT_C)
    assert total(batch, "revenue")[0] == total(dossier, "revenue")[0] == Decimal("22485.06")
    assert "DatevExtfReader" in calculate_spans(recorded, batch.run_id)
    assert "GdpduReader" in calculate_spans(recorded, dossier.run_id)
    assert identity(recorded, batch.run_id).startswith("datev:client_a@")
    assert identity(recorded, dossier.run_id).startswith("gdpdu:client_c@")
    assert recorded.spans(batch.run_id)[0].context.trace_id != recorded.spans(dossier.run_id)[0].context.trace_id


def test_the_wrong_number_runs_green_and_names_its_chart(recorded: Telemetry) -> None:
    wrong = ask(recorded, "receivables__sources", "DatevExtfReader", CLIENT_B)
    assert total(wrong, "receivables")[0] == Decimal("13750.50")
    assert all(span.status.status_code is not StatusCode.ERROR for span in recorded.spans(wrong.run_id))
    assert (
        calculate_spans(recorded, wrong.run_id)["SkrAccountFeatureGroup"].attributes["mloda.declared.chart"] == "SKR04"
    )
    assert SkrAccountFeatureGroup in {step.feature_group for step in wrong.plan}

    right = ask(recorded, "receivables__sources", "DatevExtfReader", CLIENT_B, READS_SKR03)
    assert total(right, "receivables")[0] == Decimal("7500.00")
    assert calculate_spans(recorded, right.run_id)["Skr03Accounts"].attributes["mloda.declared.chart"] == "SKR03"
    groups = {step.feature_group for step in right.plan}
    assert Skr03Accounts in groups and SkrAccountFeatureGroup not in groups


def test_a_stated_chart_turns_the_wrong_number_into_a_refusal(recorded: Telemetry) -> None:
    with pytest.raises(ChartConflict, match="state SKR03 and this host reads SKR04"):
        ask(recorded, "receivables__sources", "DatevExtfReader", CLIENT_B_CORRECTED)
    run_id = recorded.latest_run()
    assert run_id is not None
    spans = calculate_spans(recorded, run_id)
    assert list(spans) == ["DatevExtfReader", "Closing2025", "SkrAccountFeatureGroup"]
    refused = spans["SkrAccountFeatureGroup"]
    assert refused.status.status_code is StatusCode.ERROR
    assert str(refused.attributes["error.type"]).endswith("ChartConflict")
    assert [e.eventType for e in step_events(recorded, run_id)] == [RunState.COMPLETE, RunState.COMPLETE, RunState.FAIL]
    assert chain(recorded.events(run_id))[-1] == "SkrAccountFeatureGroup (failed)"

    right = ask(recorded, "receivables__sources", "DatevExtfReader", CLIENT_B_CORRECTED, READS_SKR03)
    assert total(right, "receivables")[0] == Decimal("7500.00")


def test_the_two_exports_of_client_b_differ_only_in_the_stated_chart() -> None:
    silent, stated = ((folder / BATCH).read_bytes().decode("cp1252") for folder in (CLIENT_B, CLIENT_B_CORRECTED))
    silent_header, silent_rows = silent.split("\r\n", 1)
    stated_header, stated_rows = stated.split("\r\n", 1)
    assert silent_rows == stated_rows
    pairs = zip(silent_header.split(";"), stated_header.split(";"), strict=True)
    assert [pair for pair in pairs if pair[0] != pair[1]] == [("", '"03"')]


def page(shown: Any) -> str:
    return str(shown._mime_()[1])


def test_show_formats_the_answer_the_trace_and_the_lineage(recorded: Telemetry) -> None:
    result = ask(recorded, "revenue__sources", "DatevExtfReader", CLIENT_A)
    shown = page(show(result, recorded))
    assert "revenue__sources, 22,485.06" in shown and f"{BATCH}@" in shown
    assert f"lineage run {result.run_id}" in shown and "4 of 4 lineage steps point at this trace" in shown
    assert "file datev:client_a@" in shown and "chart SKR04" in shown and "<svg" in shown

    summary = page(show(result, recorded, evidence=False))
    assert "22,485.06" in summary and "4 steps, all ok" in summary
    assert "<table" not in summary and "<svg" not in summary


def test_show_formats_a_refusal_with_the_run_that_raised_it(recorded: Telemetry) -> None:
    earlier = ask(recorded, "revenue__sources", "DatevExtfReader", CLIENT_A)
    # A refusal that no run raised gets no run's evidence, least of all the earlier run's.
    stray = page(show(ValueError("no such client"), recorded))
    assert "ValueError: no such client" in stray and "lineage run" not in stray and "<table" not in stray

    with pytest.raises(ChartConflict) as refusal:
        ask(recorded, "receivables__sources", "DatevExtfReader", CLIENT_B_CORRECTED)
    run_id = recorded.refused_run(refusal.value)
    assert run_id is not None and run_id != earlier.run_id
    assert recorded.refused_run(ValueError("another failure")) is None
    shown = page(show(refusal.value, recorded))
    assert "ChartConflict: the books state SKR03" in shown
    assert f"lineage run {run_id}" in shown and "3 of 3 lineage steps point at this trace" in shown
    assert "<td>error</td>" in shown and "refused ChartConflict" in shown


class Forward:
    """A lineage server stand-in: keeps what it is sent, or fails."""

    def __init__(self, down: bool = False) -> None:
        self.down, self.events = down, []

    def emit(self, event: Any) -> None:
        if self.down:
            raise ConnectionError("lineage server down")
        self.events.append(event)


def test_a_lineage_server_gets_every_event_and_one_that_is_down_costs_nothing(recorded: Telemetry) -> None:
    server = Forward()
    recorded.transport.forward = server
    up = ask(recorded, "revenue__sources", "DatevExtfReader", CLIENT_A)
    assert server.events == recorded.events(up.run_id) != []

    recorded.transport.forward = Forward(down=True)
    down = ask(recorded, "revenue__sources", "DatevExtfReader", CLIENT_A)
    assert recorded.transport.forward is None, "a server that failed is left alone"
    assert [e.eventType for e in step_events(recorded, down.run_id)] == [RunState.COMPLETE] * 4
    assert "4 of 4 lineage steps point at this trace" in page(show(down, recorded))
