from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from mloda.provider import FeatureSet
from mloda.user import Options

from mloda_demo.physical_ai.clip import nearest_in_corridor
from mloda_demo.physical_ai.definition import STOP_THRESHOLD_M, DepthToMetres, NearestAhead, StopRule, to_metres

FeatureSetFactory = Callable[..., FeatureSet]


@pytest.mark.parametrize(
    ("feature_group", "name"),
    [(DepthToMetres, "depth_m"), (NearestAhead, "nearest_ahead_m"), (StopRule, "stop")],
)
def test_matches_its_feature_name_only(feature_group: type, name: str) -> None:
    assert feature_group.match_feature_group_criteria(name, Options())
    assert not feature_group.match_feature_group_criteria("depth_raw", Options())


def test_to_metres_uses_the_declared_scale_and_keeps_zero_invalid() -> None:
    metres = to_metres(np.array([[2915, 0]], dtype=np.uint16), 5000.0)
    assert metres[0, 0] == pytest.approx(0.583)
    assert np.isnan(metres[0, 1])


def test_to_metres_assumes_millimetres_for_undeclared_uint16() -> None:
    assert to_metres(np.array([[2915]], dtype=np.uint16), float("nan"))[0, 0] == pytest.approx(2.915)


def test_to_metres_rejects_an_undeclared_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="int64"):
        to_metres(np.array([[1]], dtype=np.int64), float("nan"))


def test_depth_to_metres_converts_every_frame_by_its_scale(feature_set: FeatureSetFactory) -> None:
    raw = pd.Series([np.full((4, 6), 1000, dtype=np.uint16)] * 2, dtype=object)
    data = pd.DataFrame({"depth_raw": raw, "depth_scale": [np.nan, 5000.0]})
    data = DepthToMetres.calculate_feature(data, feature_set("depth_m"))
    assert data["depth_m"].iloc[0][0, 0] == pytest.approx(1.0)
    assert data["depth_m"].iloc[1][0, 0] == pytest.approx(0.2)


def test_depth_to_metres_declares_what_it_assumes() -> None:
    declared = DepthToMetres.declared_attributes(None)
    assert (declared["unit"], declared["assumes.uint16"]) == ("m", "mm")


def test_nearest_in_corridor_ignores_zeros_and_the_edges() -> None:
    image = np.full((8, 9), 5.0)
    image[:, :3] = 0.5  # outside the corridor
    image[2:6, 3:6] = [[0.0, 2.0, 3.0]] * 4  # zero means no data
    assert nearest_in_corridor(image) == pytest.approx(2.0)
    assert np.isnan(nearest_in_corridor(np.zeros((8, 9))))


def test_nearest_ahead_is_per_frame(feature_set: FeatureSetFactory) -> None:
    data = pd.DataFrame({"depth_m": pd.Series([np.full((8, 9), 2.0), np.full((8, 9), 0.5)], dtype=object)})
    data = NearestAhead.calculate_feature(data, feature_set("nearest_ahead_m"))
    assert list(data["nearest_ahead_m"]) == [2.0, 0.5]
    assert NearestAhead.declared_attributes(None)["unit"] == "m"


def test_stop_rule_threshold_is_one_metre(feature_set: FeatureSetFactory) -> None:
    data = pd.DataFrame({"nearest_ahead_m": [0.99, 1.0, float("nan")]})
    data = StopRule.calculate_feature(data, feature_set("stop"))
    assert list(data["stop"]) == [True, False, True]
    declared = StopRule.declared_attributes(None)
    assert (declared["threshold_m"], declared["on_missing"]) == (STOP_THRESHOLD_M, "stop")
