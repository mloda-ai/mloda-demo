"""Applications feature group backed by SQLite instead of JSON."""

import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from mloda.provider import BaseInputData, ComputeFramework, DataCreator, FeatureGroup, FeatureSet
from mloda.user import Index
from mloda_plugins.compute_framework.base_implementations.pandas.dataframe import PandasDataFrame

DEMO_DATA_DIR = Path(__file__).resolve().parents[3] / "demo_data"

COLUMNS = {"customer_id", "credit_amount", "purpose"}


class ApplicationsSqliteFG(FeatureGroup):
    """Applicant metadata from SQLite database."""

    compute_framework: type[ComputeFramework] = PandasDataFrame

    @classmethod
    def index_columns(cls) -> list[Index] | None:
        return [Index(("customer_id",))]

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator(COLUMNS)

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        db_path = DEMO_DATA_DIR / "applications.db"
        conn = sqlite3.connect(str(db_path))
        df = pd.read_sql_query(
            "SELECT id AS customer_id, requested_amount AS credit_amount, purpose FROM applications",
            conn,
        )
        conn.close()
        return df

    @classmethod
    def compute_framework_rule(cls) -> set[type[ComputeFramework]] | None:
        return {cls.compute_framework}
