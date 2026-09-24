import importlib.util
from pathlib import Path

import numpy as np
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


def test_closest_frame_falls_back_to_the_last_frame_without_a_valid_reading() -> None:
    result = run(TumDepth)
    blind = Run(result.label, result.reader, result.table.assign(nearest_ahead_m=float("nan")), result.spans)
    assert closest_frame(blind) == 53


def test_check_passes_the_declared_reader() -> None:
    _, nearest, stop = _chair(run(TumDepth, check=True))
    assert nearest == pytest.approx(0.583, abs=0.005)
    assert stop


@pytest.mark.parametrize(
    ("chain", "features", "nearest", "reader"),
    [
        (welded.KINECT_CHAIN, welded.KINECT_FEATURES, "kinect_nearest_m", DepthPng),
        (welded.TUM_CHAIN, welded.TUM_FEATURES, "tum_nearest_m", TumDepth),
    ],
)
def test_welded_chains_give_the_same_numbers_as_the_shared_definition(
    chain: frozenset[type], features: tuple[str, ...], nearest: str, reader: type
) -> None:
    result = run(None, features=features, feature_groups=chain)
    assert result.passed
    np.testing.assert_allclose(result.table[nearest].to_numpy(), run(reader).table["nearest_ahead_m"].to_numpy())


def test_slides_script_draws_both_pictures(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts" / "physical_ai_slides.py"
    spec = importlib.util.spec_from_file_location("physical_ai_slides", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(tmp_path)
    assert {path.name for path in tmp_path.glob("*.svg")} == {"pipelines_welded.svg", "pipeline_shared.svg"}
