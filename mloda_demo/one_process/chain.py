"""Shared bits of every chain in the one-process demo."""

from __future__ import annotations

from mloda.provider import ComputeFramework
from mloda.user import FeatureName
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame


class PandasOnly:
    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {PandasDataFrame}


def source_of(feature_name: FeatureName | str) -> str:
    """The chained name without its last step: tum__metres__nearest -> tum__metres."""
    return str(feature_name).rsplit("__", 1)[0]


def root_of(feature_name: FeatureName | str) -> str:
    """The first step of a chained name: finance__revenue__per_customer -> finance."""
    return str(feature_name).split("__", 1)[0]
