from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from mloda.provider import FeatureSet
from mloda.user import Options

from mloda_demo.physical_ai.definition import (
    CX,
    FX,
    Approaching,
    BrakeRule,
    Calibration,
    DepthToMetres,
    NearestDistance,
)

FeatureSetFactory = Callable[..., FeatureSet]


@pytest.mark.parametrize(
    ("feature_group", "name"),
    [
        (DepthToMetres, "depth_m"),
        (Calibration, "x_m"),
        (Calibration, "y_m"),
        (NearestDistance, "nearest_distance_m"),
        (BrakeRule, "brake"),
        (Approaching, "approaching"),
    ],
)
def test_matches_its_feature_name_only(feature_group: type, name: str) -> None:
    assert feature_group.match_feature_group_criteria(name, Options())
    assert not feature_group.match_feature_group_criteria("depth_raw", Options())


@pytest.mark.parametrize(
    ("raw", "fault", "expected"),
    [
        (np.array([1.2], dtype=np.float32), False, 1.2),
        (np.array([1.2], dtype=np.float32), True, 1.2),
        (np.array([1200], dtype=np.uint16), False, 1.2),
        (np.array([1200], dtype=np.uint16), True, 12.0),
    ],
)
def test_depth_to_metres_assumes_unit_per_encoding(
    feature_set: FeatureSetFactory, raw: np.ndarray, fault: bool, expected: float
) -> None:
    data = DepthToMetres.calculate_feature(pd.DataFrame({"depth_raw": raw}), feature_set("depth_m", fault=fault))
    assert data["depth_m"].iloc[0] == pytest.approx(expected, abs=1e-6)


def test_depth_to_metres_rejects_unknown_encoding(feature_set: FeatureSetFactory) -> None:
    with pytest.raises(ValueError, match="int64"):
        DepthToMetres.calculate_feature(
            pd.DataFrame({"depth_raw": np.array([1], dtype=np.int64)}), feature_set("depth_m")
        )


def test_depth_to_metres_declares_what_it_assumes(feature_set: FeatureSetFactory) -> None:
    assert DepthToMetres.declared_attributes(feature_set("depth_m", fault=True))["assumes.uint16"] == "cm"
    assert DepthToMetres.declared_attributes(feature_set("depth_m"))["assumes.uint16"] == "mm"
    assert DepthToMetres.declared_attributes(None)["unit"] == "m"


def test_calibration_maps_camera_optical_to_base_link(feature_set: FeatureSetFactory) -> None:
    data = pd.DataFrame({"depth_m": [2.0, 2.0], "u": [CX, CX - FX]})
    data = Calibration.calculate_feature(data, feature_set("x_m"))
    assert list(data["x_m"]) == [2.0, 2.0]
    assert list(data["y_m"]) == pytest.approx([0.0, 2.0])


def test_nearest_distance_is_the_minimum_per_frame_and_object(feature_set: FeatureSetFactory) -> None:
    data = pd.DataFrame(
        {"x_m": [3.0, 4.0, 3.0, 1.0], "y_m": [4.0, 0.0, 0.0, 0.0], "frame": [0, 0, 1, 0], "object_id": [1, 1, 1, 2]}
    )
    data = NearestDistance.calculate_feature(data, feature_set("nearest_distance_m"))
    assert list(data["nearest_distance_m"]) == [4.0, 4.0, 3.0, 1.0]


def test_brake_rule_threshold_is_two_metres(feature_set: FeatureSetFactory) -> None:
    data = BrakeRule.calculate_feature(pd.DataFrame({"nearest_distance_m": [1.99, 2.0]}), feature_set("brake"))
    assert list(data["brake"]) == [True, False]


def test_approaching_uses_closing_speed_since_previous_frame(feature_set: FeatureSetFactory) -> None:
    data = pd.DataFrame(
        {
            "object_id": [1, 1, 2, 2],
            "frame": [0, 1, 0, 1],
            "timestamp": [0.0, 0.2, 0.0, 0.2],
            "nearest_distance_m": [6.0, 5.6, 4.0, 3.95],
        }
    )
    data = Approaching.calculate_feature(data, feature_set("approaching"))
    assert list(data["approaching"]) == [False, True, False, False]
