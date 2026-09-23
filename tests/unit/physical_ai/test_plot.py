import pandas as pd

from mloda_demo.physical_ai.plot import RANGE_M, top_down


def test_objects_beyond_the_axes_sit_on_the_edge_with_their_distance() -> None:
    points = pd.DataFrame(
        {"x_m": [1.2, 42.0], "y_m": [0.0, 0.0], "brake": [True, False]}, index=pd.Index([7, 5], name="object_id")
    )
    axes = top_down(points, ghost=(1.2, 0.0)).axes[0]
    assert axes.get_ylim() == (0.0, RANGE_M)
    labels = [text.get_text() for text in axes.texts]
    assert "7" in labels
    assert "5: 42 m" in labels
    edge = [offsets for collection in axes.collections for offsets in collection.get_offsets()]
    assert all(0.0 <= y <= RANGE_M for _, y in edge)
