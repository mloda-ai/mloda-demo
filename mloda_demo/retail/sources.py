"""The shop's order lines, read by finance's ledger or by marketing's export. Each reader declares what it delivers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from mloda.provider import (
    INPUT_DATA_STAGE,
    ClaimRoute,
    NamePolicy,
    SourceFeatureGroup,
    SourceMatch,
    record_match_rejection,
)
from mloda.user import DataAccessCollection, Options

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR
from mloda_demo.pandas_only import PandasOnly

LEDGER = DEMO_DATA_DIR / "retail" / "ledger.csv.gz"
COLUMNS = ("invoice", "customer_id", "invoice_date", "quantity", "price")
CANCELLED = "C"  # invoice prefix of a cancellation

# Group options travel from a requested feature down to the reader.
AS_OF = "as_of"
NEEDS = "needs"  # what a definition needs its source to deliver
CUSTOMER = "customer"  # one customer's order lines only
# Options naming the source class and its file (a class-keyed pointer cannot ride on a feature_group= request).
ORDER_SOURCE = "order_source"
ORDERS = "orders"


class OrderSource(PandasOnly, SourceFeatureGroup):
    """Order lines from a CSV file; a subclass says what it delivers."""

    CLAIM_ROUTES = (ClaimRoute("file", NamePolicy.DECLARED, True),)

    delivers: ClassVar[frozenset[str]] = frozenset()
    owner: ClassVar[str] = ""
    label: ClassVar[str] = ""

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        return frozenset(COLUMNS)

    @classmethod
    def find_sources(
        cls, route: ClaimRoute, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> list[SourceMatch]:
        path = options.get(ORDERS)
        chosen = options.get(ORDER_SOURCE) == cls.get_class_name()
        if not chosen or not isinstance(path, (str, Path)) or not Path(path).is_file():
            return []
        needs = options.get(NEEDS)
        if needs is not None and needs not in cls.delivers:
            record_match_rejection(
                cls.get_class_name(),
                f"needs {needs}; {cls.label} delivers {' and '.join(sorted(cls.delivers))} only",
                stage=INPUT_DATA_STAGE,
            )
            return []
        return [SourceMatch(source=str(path), access=Path(path))]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        lines = pd.read_csv(match.access, dtype={"invoice": str}, parse_dates=["invoice_date"])
        customer = features.get_options_key(CUSTOMER)
        if customer is not None:
            lines = lines[lines["customer_id"] == int(customer)]
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
