"""The ML chain: one monthly payment, over every past credit and for one new request."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR
from mloda_demo.one_process.chain import PandasOnly, source_of

HISTORY = DEMO_DATA_DIR / "german_credit.csv"
APPLICANT = DEMO_DATA_DIR / "one_process" / "applicant.json"


class History(PandasOnly, FeatureGroup):
    """Offline: every past credit in the UCI German credit table."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"history_amount", "history_months"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        table = pd.read_csv(HISTORY, usecols=["credit_amount", "duration"])
        return pd.DataFrame({"history_amount": table["credit_amount"], "history_months": table["duration"]})


class Applicant(PandasOnly, FeatureGroup):
    """Online: one request, as it arrives."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"applicant_amount", "applicant_months"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        request = json.loads(APPLICANT.read_text())
        return pd.DataFrame({"applicant_amount": [request["amount"]], "applicant_months": [request["months"]]})


class MonthlyPayment(PandasOnly, FeatureChainParserMixin, FeatureGroup):
    """Amount over months: the same number in training and for the request."""

    PREFIX_PATTERN = r"^.+__monthly_payment$"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        source = source_of(feature_name)
        return {Feature(f"{source}_amount"), Feature(f"{source}_months")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source = source_of(feature.name)
            data[str(feature.name)] = data[f"{source}_amount"] / data[f"{source}_months"]
        return data
