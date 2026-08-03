"""Root FeatureGroup: read pruned applications.json and emit per-customer columns.

Columns emitted:
    customer_id       string  (renamed from the source "id" field)
    credit_amount     int     (renamed from "requested_amount" to match German Credit)
    purpose           string  (already aligned to German Credit categories)
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR

COLUMNS = {"customer_id", "credit_amount", "purpose"}


class ApplicationsFG(FeatureGroup):
    """Structured applicant metadata loaded from applications.json."""

    compute_framework: type[ComputeFramework] = PandasDataFrame

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("customer_id",))]

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(COLUMNS)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        path = DEMO_DATA_DIR / "applications.json"
        records = json.loads(path.read_text())
        df = pd.DataFrame(
            {
                "customer_id": [r["id"] for r in records],
                "credit_amount": [int(r["requested_amount"]) for r in records],
                "purpose": [r["purpose"] for r in records],
            }
        )
        return df

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}
