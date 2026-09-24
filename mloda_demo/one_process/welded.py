"""Two robot pipelines that share no step: the picture the talk starts from."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from mloda_demo.one_process.chain import PandasOnly
from mloda_demo.one_process.robot import redwood_frames, to_metres, tum_frames
from mloda_demo.physical_ai.clip import nearest_in_corridor


class TumFrames(PandasOnly, FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"tum_depth"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"tum_depth": pd.Series(list(tum_frames()), dtype=object)})


class TumMetres(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tum_depth")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tum_metres"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        metres = [to_metres(raw, 5000) for raw in data["tum_depth"]]
        data["tum_metres"] = pd.Series(metres, index=data.index, dtype=object)
        return data


class TumNearest(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tum_metres")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tum_nearest"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["tum_nearest"] = [nearest_in_corridor(metres) for metres in data["tum_metres"]]
        return data


class TumStop(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("tum_nearest")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"tum_stop"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["tum_stop"] = ~(data["tum_nearest"] >= 1.0)
        return data


class RedwoodFrames(PandasOnly, FeatureGroup):
    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"redwood_depth"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame({"redwood_depth": pd.Series(list(redwood_frames()), dtype=object)})


class RedwoodMetres(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("redwood_depth")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"redwood_metres"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        metres = [to_metres(raw, 1000) for raw in data["redwood_depth"]]
        data["redwood_metres"] = pd.Series(metres, index=data.index, dtype=object)
        return data


class RedwoodNearest(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("redwood_metres")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"redwood_nearest"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["redwood_nearest"] = [nearest_in_corridor(metres) for metres in data["redwood_metres"]]
        return data


class RedwoodStop(PandasOnly, FeatureGroup):
    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature("redwood_nearest")}

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return {"redwood_stop"}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["redwood_stop"] = ~(data["redwood_nearest"] >= 1.0)
        return data
