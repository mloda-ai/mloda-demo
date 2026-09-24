"""Two welded pipelines that share no plugin: the runs the first pipeline picture is drawn from."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options
from numpy.typing import NDArray

from mloda_demo.physical_ai.clip import CLIP_DIR, frame_index, load_depth, nearest_in_corridor
from mloda_demo.physical_ai.definition import PandasOnly


def _frames(column: str) -> pd.DataFrame:
    index = frame_index(CLIP_DIR)
    depth = [load_depth(CLIP_DIR / path) for path in index["depth"]]
    return pd.DataFrame({"frame": index["frame"].to_numpy(), column: pd.Series(depth, dtype=object)})


def _metres(raw: NDArray[Any], scale: float) -> NDArray[np.float64]:
    metres = raw.astype(np.float64) / scale
    metres[raw == 0] = np.nan
    return metres


class KinectDepthPng(PandasOnly, FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"frame", "kinect_depth_raw"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _frames("kinect_depth_raw")


class KinectToMetres(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("kinect_depth_raw")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"kinect_depth_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        metres = [_metres(raw, 1000.0) for raw in data["kinect_depth_raw"]]
        data["kinect_depth_m"] = pd.Series(metres, index=data.index, dtype=object)
        return data


class KinectNearest(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("kinect_depth_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"kinect_nearest_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["kinect_nearest_m"] = [nearest_in_corridor(metres) for metres in data["kinect_depth_m"]]
        return data


class KinectStop(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("kinect_nearest_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"kinect_stop"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["kinect_stop"] = ~(data["kinect_nearest_m"] >= 1.0)
        return data


class TumDepthPng(PandasOnly, FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"frame", "tum_depth_raw"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return _frames("tum_depth_raw")


class TumToMetres(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tum_depth_raw")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tum_depth_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        metres = [_metres(raw, 5000.0) for raw in data["tum_depth_raw"]]
        data["tum_depth_m"] = pd.Series(metres, index=data.index, dtype=object)
        return data


class TumNearest(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tum_depth_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tum_nearest_m"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["tum_nearest_m"] = [nearest_in_corridor(metres) for metres in data["tum_depth_m"]]
        return data


class TumStop(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tum_nearest_m")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tum_stop"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["tum_stop"] = ~(data["tum_nearest_m"] >= 1.0)
        return data


KINECT_CHAIN: frozenset[type[FeatureGroup]] = frozenset({KinectDepthPng, KinectToMetres, KinectNearest, KinectStop})
KINECT_FEATURES = ("frame", "kinect_nearest_m", "kinect_stop")
TUM_CHAIN: frozenset[type[FeatureGroup]] = frozenset({TumDepthPng, TumToMetres, TumNearest, TumStop})
TUM_FEATURES = ("frame", "tum_nearest_m", "tum_stop")
