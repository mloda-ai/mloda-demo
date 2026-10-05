"""A stand-in feature store: `net_spend_30d` as the batch stored it, looked up by customer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet

from mloda_demo.retail.kpis import Kpi
from mloda_demo.retail.sources import CUSTOMER


@dataclass
class Store:
    values: dict[int, float] = field(default_factory=dict)
    version: str = ""  # of the definition that filled it
    as_of: str = ""

    def fill(self, values: dict[int, float], version: str, as_of: str) -> None:
        self.values, self.version, self.as_of = values, version, as_of

    def clear(self) -> None:
        self.fill({}, "", "")


STORE = Store()


class FeatureStore(Kpi, FeatureGroup):
    """Stand-in feature store: what the batch stored, looked up by customer."""

    NAMES = ("net_spend_30d",)
    OWNER = "store"

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"net_spend_30d"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        customer = int(features.get_options_key(CUSTOMER))
        return pd.DataFrame({"net_spend_30d": [STORE.values[customer]]})
