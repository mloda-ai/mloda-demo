"""The shop's order lines, read by finance's ledger or by marketing's export. Each reader declares what it delivers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from mloda.provider import INPUT_DATA_STAGE, BaseInputData, FeatureGroup, FeatureSet, record_match_rejection
from mloda.user import Options

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR
from mloda_demo.pandas_only import PandasOnly

LEDGER = DEMO_DATA_DIR / "retail" / "ledger.csv.gz"
COLUMNS = ("invoice", "customer_id", "invoice_date", "quantity", "price")
CANCELLED = "C"  # invoice prefix of a cancellation

# Group options travel from a requested feature down to the reader.
AS_OF = "as_of"
NEEDS = "needs"  # what a definition needs its source to deliver


class OrderSource(BaseInputData):
    """Order lines from a CSV file; a subclass says what it delivers."""

    delivers: ClassVar[frozenset[str]] = frozenset()
    owner: ClassVar[str] = ""
    label: ClassVar[str] = ""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if not isinstance(data_access, (str, Path)) or not Path(data_access).is_file():
            return None
        if not set(COLUMNS).issuperset(feature_names):
            return None
        needs = options.get(NEEDS)
        if needs is not None and needs not in cls.delivers:
            record_match_rejection(
                cls.get_class_name(),
                f"needs {needs}; {cls.label} delivers {' and '.join(sorted(cls.delivers))} only",
                stage=INPUT_DATA_STAGE,
            )
            return None
        return data_access

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        lines = pd.read_csv(data_access, dtype={"invoice": str}, parse_dates=["invoice_date"])
        return cls.keep(lines).reset_index(drop=True)

    @classmethod
    def keep(cls, lines: pd.DataFrame) -> pd.DataFrame:
        return lines


class ShopLedger(OrderSource):
    """Finance's ledger: every order and every cancellation."""

    delivers = frozenset({"orders", "cancellations"})
    owner = "finance"
    label = "the shop ledger"


class MarketingExport(OrderSource):
    """Marketing's export: the same orders, cancellations left out."""

    delivers = frozenset({"orders"})
    owner = "marketing"
    label = "the marketing export"

    @classmethod
    def keep(cls, lines: pd.DataFrame) -> pd.DataFrame:
        return lines[~lines["invoice"].str.startswith(CANCELLED)]


class Orders(PandasOnly, FeatureGroup):
    """The order lines; a group option keyed by the reader class picks the source."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return OrderSource()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return OrderSource().load(features)
