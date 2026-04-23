"""Root FeatureGroup: read synthetic financial_overview_<id>.xlsx per customer.

Each xlsx has two columns (field, value) on sheet "financials". Fields map 1:1 to
numeric German Credit features (duration, installment_commitment, residence_since,
age, existing_credits, num_dependents).
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Type

import pandas as pd

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR

COLUMNS = {
    "customer_id",
    "duration",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
}


class FinancialsFG(FeatureGroup):
    """Numeric financial fields extracted from per-customer Excel overviews."""

    compute_framework: Type[ComputeFramework] = PandasDataFrame

    @classmethod
    def index_columns(cls) -> Optional[List[Index]]:
        return [Index(("customer_id",))]

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(COLUMNS)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        rows = []
        for xlsx_path in sorted(DEMO_DATA_DIR.glob("financial_overview_*.xlsx")):
            customer_id = xlsx_path.stem.replace("financial_overview_", "")
            sheet = pd.read_excel(xlsx_path, sheet_name="financials", engine="openpyxl")
            row: dict[str, Any] = {"customer_id": customer_id}
            for _, record in sheet.iterrows():
                row[str(record["field"])] = int(record["value"])
            rows.append(row)
        return pd.DataFrame(rows, columns=list(COLUMNS))

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {cls.compute_framework}
