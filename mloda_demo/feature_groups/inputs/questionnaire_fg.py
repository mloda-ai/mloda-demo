"""Root FeatureGroup: parse synthetic qa_<id>.md per customer.

Each markdown file has sections with "- key: value" lines. We parse those into
columns matching the categorical German Credit features.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Set, Type

import pandas as pd

from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR

CATEGORICAL_FIELDS = [
    "checking_status",
    "credit_history",
    "savings_status",
    "employment",
    "personal_status",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
    "foreign_worker",
]

COLUMNS = {"customer_id", *CATEGORICAL_FIELDS}

_LINE = re.compile(r"^-\s+([a-z_]+)\s*:\s*(.+?)\s*$")


def _parse_md(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class QuestionnaireFG(FeatureGroup):
    """Categorical fields extracted from per-customer Q&A markdown files."""

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
        for md_path in sorted(DEMO_DATA_DIR.glob("qa_*.md")):
            customer_id = md_path.stem.replace("qa_", "")
            parsed = _parse_md(md_path.read_text())
            row: dict[str, Any] = {"customer_id": customer_id}
            for field in CATEGORICAL_FIELDS:
                row[field] = parsed.get(field, "")
            rows.append(row)
        return pd.DataFrame(rows, columns=["customer_id", *CATEGORICAL_FIELDS])

    @classmethod
    def compute_framework_rule(cls) -> Optional[Set[Type[ComputeFramework]]]:
        return {cls.compute_framework}
