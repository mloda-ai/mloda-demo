from collections.abc import Callable
from typing import Any

import pytest
from mloda.provider import FeatureSet
from mloda.user import Feature, Options


@pytest.fixture
def feature_set() -> Callable[..., FeatureSet]:
    def make(name: str, **group: Any) -> FeatureSet:
        features = FeatureSet()
        features.add(Feature(name, Options(group=group)))
        return features

    return make
