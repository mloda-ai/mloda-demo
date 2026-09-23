import io

import matplotlib
import pandas as pd
from matplotlib import font_manager

from mloda_demo.physical_ai.plot import RANGE_M, top_down

POINTS = pd.DataFrame(
    {"x_m": [1.2, 42.0], "y_m": [0.0, 0.0], "brake": [True, False]}, index=pd.Index([7, 5], name="object_id")
)


def test_objects_beyond_the_axes_sit_on_the_edge_with_their_distance() -> None:
    axes = top_down(POINTS, ghost=(1.2, 0.0)).axes[0]
    assert axes.get_ylim() == (0.0, RANGE_M)
    labels = [text.get_text() for text in axes.texts]
    assert "7" in labels
    assert "5: 42 m" in labels
    edge = [offsets for collection in axes.collections for offsets in collection.get_offsets()]
    assert all(0.0 <= y <= RANGE_M for _, y in edge)


def test_plot_text_uses_the_shipped_brand_font_when_drawn_later() -> None:
    figure = top_down(POINTS, title="device B")
    figure.savefig(io.BytesIO(), format="png")
    axes = figure.axes[0]
    texts = [axes.title, axes.xaxis.label, *axes.get_xticklabels(), *axes.texts]
    paths = {font_manager.findfont(text.get_fontproperties(), fallback_to_default=False) for text in texts}
    assert {font_manager.get_font(path).family_name for path in paths} == {"Schibsted Grotesk"}
    assert "Schibsted Grotesk" not in matplotlib.rcParams["font.family"]
