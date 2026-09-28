import numpy as np
import pytest
from matplotlib.axes import Axes

from mloda_demo.physical_ai.plot import ATTRIBUTION, frame_view, pipeline_graph


def _labels(axes: Axes) -> list[str]:
    return [text.get_text() for text in axes.texts if text.get_text()]


def test_frame_view_shows_the_number_and_the_verdict() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    depth = np.full((480, 640), 2.0)
    texts = [text.get_text() for text in frame_view(rgb, depth, 0.58, True, 15.8).texts]
    assert "nearest ahead: 0.58 m" in texts
    assert "STOP" in texts
    assert ATTRIBUTION in texts
    assert "go" in [text.get_text() for text in frame_view(rgb, depth, 2.92, False, 15.8).texts]


def test_pipeline_graph_draws_shared_steps_once() -> None:
    chains = [["A", "Shared1", "Shared2"], ["B", "Shared1", "Shared2"]]
    separate = _labels(pipeline_graph([["A", "A2"], ["B", "B2"]]).axes[0])
    merged = _labels(pipeline_graph(chains).axes[0])
    assert sorted(separate) == ["A", "A2", "B", "B2"]
    assert merged.count("Shared1") == 1
    assert sorted(merged) == ["A", "B", "Shared1", "Shared2"]
    with pytest.raises(ValueError, match="step of its own"):
        pipeline_graph([["Shared1"], ["Shared1"]])
