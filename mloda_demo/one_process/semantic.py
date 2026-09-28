"""The semantic chain: one definition of revenue over two departments' exports."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR
from mloda_demo.one_process.chain import PandasOnly, root_of, source_of

FINANCE = DEMO_DATA_DIR / "one_process" / "finance.csv"
SALES = DEMO_DATA_DIR / "one_process" / "sales.csv"


class Finance(PandasOnly, FeatureGroup):
    """Finance's invoices: the amount billed and what was refunded."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"finance_customer", "finance_gross", "finance_deductions"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        invoices = pd.read_csv(FINANCE)
        return pd.DataFrame(
            {
                "finance_customer": invoices["customer"],
                "finance_gross": invoices["amount"],
                "finance_deductions": invoices["refunded"],
            }
        )


class Sales(PandasOnly, FeatureGroup):
    """Sales' deals: what was booked and the discount given."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sales_customer", "sales_gross", "sales_deductions"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        deals = pd.read_csv(SALES)
        return pd.DataFrame(
            {"sales_customer": deals["customer"], "sales_gross": deals["booked"], "sales_deductions": deals["discount"]}
        )


class Revenue(PandasOnly, FeatureChainParserMixin, FeatureGroup):
    """What the customer actually pays: gross minus deductions, whichever department booked it."""

    PREFIX_PATTERN = r"^.+__revenue$"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        source = source_of(feature_name)
        return {Feature(f"{source}_gross"), Feature(f"{source}_deductions")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source = source_of(feature.name)
            data[str(feature.name)] = data[f"{source}_gross"] - data[f"{source}_deductions"]
        return data


class PerCustomer(PandasOnly, FeatureChainParserMixin, FeatureGroup):
    """Each row's customer total."""

    PREFIX_PATTERN = r"^.+__per_customer$"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {Feature(source_of(feature_name)), Feature(f"{root_of(feature_name)}_customer")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            customer = data[f"{root_of(feature.name)}_customer"]
            data[str(feature.name)] = data[source_of(feature.name)].groupby(customer).transform("sum")
        return data
