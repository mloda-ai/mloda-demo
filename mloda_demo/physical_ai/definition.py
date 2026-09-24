"""The shared definition: one meaning of the nearest obstacle ahead, whichever reader delivered the depth."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mloda.provider import ComputeFramework, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame
from numpy.typing import NDArray

from mloda_demo.physical_ai.clip import CORRIDOR, PERCENTILE, nearest_in_corridor

STOP_THRESHOLD_M = 1.0
# What the conversion assumes when the reader declares no scale: 16-bit depth in millimetres.
ASSUMED_SCALE = {"uint16": 1000.0}
ASSUMED_UNIT = {"uint16": "mm"}


class PandasOnly:
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}


def to_metres(raw: NDArray[Any], scale: float) -> NDArray[np.float64]:
    """Divide by the declared scale, or by the assumed one when undeclared (NaN); zero stays invalid as NaN."""
    if np.isnan(scale):
        encoding = str(raw.dtype)
        if encoding not in ASSUMED_SCALE:
            raise ValueError(f"DepthToMetres assumes no scale for {encoding} depth")
        scale = ASSUMED_SCALE[encoding]
    metres = raw.astype(np.float64) / scale
    metres[raw == 0] = np.nan
    return metres


class DepthToMetres(PandasOnly, FeatureGroup):
    """Raw depth to metres by the reader's declared scale."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("depth_raw"), Feature("depth_scale")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"depth_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        metres = [to_metres(raw, scale) for raw, scale in zip(data["depth_raw"], data["depth_scale"], strict=True)]
        data["depth_m"] = pd.Series(metres, index=data.index, dtype=object)
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        return {"unit": "m", **{f"assumes.{encoding}": unit for encoding, unit in ASSUMED_UNIT.items()}}


class NearestAhead(PandasOnly, FeatureGroup):
    """Nearest valid reading in the central corridor, per frame."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("depth_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"nearest_ahead_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["nearest_ahead_m"] = [nearest_in_corridor(metres) for metres in data["depth_m"]]
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        return {"unit": "m", "corridor": CORRIDOR, "percentile": f"{PERCENTILE:g}"}


class StopRule(PandasOnly, FeatureGroup):
    """Stop when the nearest obstacle ahead is closer than the threshold, or unknown."""

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("nearest_ahead_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"stop"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["stop"] = ~(data["nearest_ahead_m"] >= STOP_THRESHOLD_M)
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, float | str]:
        return {"threshold_m": STOP_THRESHOLD_M, "on_missing": "stop"}
