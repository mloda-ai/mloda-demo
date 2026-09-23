"""The shared definition: one meaning of distance, whichever device or context the data comes from."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

# Robot calibration: pinhole intrinsics, camera at the base_link origin.
CALIBRATION_VERSION = "2026-09-01"
FX = 525.0
CX = 320.0

BRAKE_THRESHOLD_M = 2.0
APPROACH_SPEED_MPS = 0.5

TO_METRES = {"m": 1.0, "cm": 0.01, "mm": 0.001}
# Unit assumed per raw depth encoding.
ASSUMED_UNITS = {"float32": "m", "uint16": "mm"}
# The bug: integer depth read as centimetres.
FAULTY_UNITS = {**ASSUMED_UNITS, "uint16": "cm"}


class PandasOnly:
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}


class DepthToMetres(PandasOnly, FeatureGroup):
    """Raw depth to metres, with the unit assumed from the raw encoding."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("depth_raw")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"depth_m"}

    @classmethod
    def assumed_units(cls, fault: bool = False) -> dict[str, str]:
        """The unit this conversion assumes per raw encoding; the check beat enforces it at the reader."""
        return FAULTY_UNITS if fault else ASSUMED_UNITS

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        raw = data["depth_raw"]
        units = cls.assumed_units(bool(features.get_options_key("fault")))
        encoding = str(raw.dtype)
        if encoding not in units:
            raise ValueError(f"DepthToMetres assumes no unit for {encoding} depth")
        data["depth_m"] = raw.astype("float64") * TO_METRES[units[encoding]]
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        fault = bool(features is not None and features.get_options_key("fault"))
        return {"unit": "m", **{f"assumes.{encoding}": unit for encoding, unit in cls.assumed_units(fault).items()}}


class Calibration(PandasOnly, FeatureGroup):
    """Camera optical frame (x right, z forward) to the base_link ground plane (x forward, y left)."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("depth_m"), Feature("u")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"x_m", "y_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["x_m"] = data["depth_m"]
        data["y_m"] = -(data["u"] - CX) * data["depth_m"] / FX
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        return {"calibration": CALIBRATION_VERSION, "source_frame": "camera_optical", "frame": "base_link"}


class NearestDistance(PandasOnly, FeatureGroup):
    """Ground-plane distance to each object's nearest point, per frame."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("x_m"), Feature("y_m"), Feature("frame"), Feature("object_id")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"nearest_distance_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        distance = pd.Series(np.hypot(data["x_m"], data["y_m"]), index=data.index)
        data["nearest_distance_m"] = distance.groupby([data["frame"], data["object_id"]]).transform("min")
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        return {"unit": "m", "frame": "base_link"}


class BrakeRule(PandasOnly, FeatureGroup):
    """Brake when an object is nearer than the threshold."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("nearest_distance_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"brake"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["brake"] = data["nearest_distance_m"] < BRAKE_THRESHOLD_M
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, float]:
        return {"threshold_m": BRAKE_THRESHOLD_M}


class Approaching(PandasOnly, FeatureGroup):
    """An object closes in faster than the threshold speed since the previous frame."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("nearest_distance_m"), Feature("timestamp"), Feature("frame"), Feature("object_id")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"approaching"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        keys = ["object_id", "frame"]
        per_frame = data.groupby(keys, as_index=False)[["timestamp", "nearest_distance_m"]].first()
        per_frame = per_frame.sort_values(["object_id", "timestamp"])
        by_object = per_frame.groupby("object_id")
        speed = -by_object["nearest_distance_m"].diff() / by_object["timestamp"].diff()
        per_frame["approaching"] = speed > APPROACH_SPEED_MPS
        data["approaching"] = (
            data[keys].merge(per_frame[[*keys, "approaching"]], on=keys, how="left")["approaching"].to_numpy()
        )
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, float]:
        return {"min_speed_mps": APPROACH_SPEED_MPS}
