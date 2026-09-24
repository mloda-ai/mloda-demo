import pytest

from mloda_demo.physical_ai import welded
from mloda_demo.physical_ai.readers import DepthPng, TumDepth
from mloda_demo.physical_ai.runner import Run, at_frame, closest_frame, raw_nearest, run

CHAIR_FRAME = 53
REJECTION = "DepthToMetres needs a declared scale; DepthPng delivers uint16 without one"


def _chair(result: Run) -> tuple[int, float, bool]:
    frame = closest_frame(result)
    row = at_frame(result, frame)
    return frame, float(row["nearest_ahead_m"]), bool(row["stop"])


def test_generic_reader_puts_the_chair_at_three_metres_and_never_stops() -> None:
    result = run(DepthPng)
    assert result.passed
    frame, nearest, stop = _chair(result)
    assert (frame, stop) == (CHAIR_FRAME, False)
    assert nearest == pytest.approx(2.915, abs=0.005)
    assert not result.table["stop"].any()
    assert raw_nearest(result, CHAIR_FRAME) == 2915


def test_tum_reader_stops_for_the_chair_through_the_same_definition() -> None:
    result = run(TumDepth)
    frame, nearest, stop = _chair(result)
    assert (frame, stop) == (CHAIR_FRAME, True)
    assert nearest == pytest.approx(0.583, abs=0.005)
    stopped = result.table.loc[result.table["stop"], "frame"]
    assert (int(stopped.min()), int(stopped.max())) == (46, 53)


def test_check_refuses_the_undeclared_reader_before_any_number() -> None:
    result = run(DepthPng, check=True)
    assert result.error == REJECTION
    assert result.table.empty
    assert not result.passed
    assert not [span for span in result.spans if span.name == "mloda.load"]


def test_check_passes_the_declared_reader() -> None:
    _, nearest, stop = _chair(run(TumDepth, check=True))
    assert nearest == pytest.approx(0.583, abs=0.005)
    assert stop


@pytest.mark.parametrize(
    ("chain", "features", "nearest", "expected"),
    [
        (welded.KINECT_CHAIN, welded.KINECT_FEATURES, "kinect_nearest_m", 2.915),
        (welded.TUM_CHAIN, welded.TUM_FEATURES, "tum_nearest_m", 0.583),
    ],
)
def test_welded_chains_give_the_same_numbers_without_sharing_a_plugin(
    chain: frozenset[type], features: tuple[str, ...], nearest: str, expected: float
) -> None:
    result = run(None, features=features, feature_groups=chain)
    assert result.passed
    assert result.table[nearest].min() == pytest.approx(expected, abs=0.005)
    assert not (welded.KINECT_CHAIN & welded.TUM_CHAIN)
