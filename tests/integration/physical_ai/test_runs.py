import pytest

from mloda_demo.physical_ai.readers import DepthReaderA, DepthReaderB
from mloda_demo.physical_ai.runner import (
    FEATURES,
    Run,
    all_passed,
    approaching_objects,
    compare,
    n_table,
    nearest_points,
    run,
)


def _object_7(result: Run) -> tuple[float, bool]:
    point = nearest_points(result).loc[7]
    return float(point["nearest_distance_m"]), bool(point["brake"])


def test_offline_device_a_brakes_for_object_7_at_1_2_m() -> None:
    distance, brake = _object_7(run(DepthReaderA))
    assert distance == pytest.approx(1.2, abs=1e-6)
    assert brake


@pytest.mark.parametrize("frame", [0, 1, 2])
def test_online_replay_agrees_with_offline_on_every_frame(frame: int) -> None:
    result = run(DepthReaderA, label="online", replay_until=frame)
    assert int(result.table["frame"].max()) == frame
    distance, brake = _object_7(result)
    assert distance == pytest.approx(1.2, abs=1e-6)
    assert brake


def test_device_b_gives_the_same_answer_through_the_same_definition() -> None:
    distance, brake = _object_7(run(DepthReaderB))
    assert distance == pytest.approx(1.2, abs=1e-6)
    assert brake


def test_fault_reads_device_b_millimetres_as_centimetres() -> None:
    result = run(DepthReaderB, fault=True)
    assert result.passed
    distance, brake = _object_7(result)
    assert distance == pytest.approx(12.0)
    assert not brake


def test_n_table_shows_the_disagreement() -> None:
    runs = compare(DepthReaderB, fault=True, frame=2)
    table = n_table(runs)
    assert all_passed(runs.values())
    assert list(table.columns) == ["offline", "online", "device B"]
    assert list(table.loc["object 7"]) == ["1.2 m", "1.2 m", "12 m"]
    assert list(table.loc["brake"]) == ["yes", "yes", "NO"]


def test_check_fails_the_broken_setup_before_any_number() -> None:
    result = run(DepthReaderB, fault=True, check_units=True)
    assert result.error == "DepthToMetres assumes uint16 depth in cm, but DepthReaderB delivers mm"
    assert result.table.empty
    assert not all_passed([result])


def test_check_passes_the_fixed_setup() -> None:
    distance, brake = _object_7(run(DepthReaderB, check_units=True))
    assert distance == pytest.approx(1.2, abs=1e-6)
    assert brake


@pytest.mark.parametrize(("reader", "fault"), [(DepthReaderA, False), (DepthReaderB, True)])
def test_approaching_names_object_3_on_every_device(reader: type, fault: bool) -> None:
    result = run(reader, fault=fault, features=(*FEATURES, "approaching"))
    assert approaching_objects(result) == [3]
