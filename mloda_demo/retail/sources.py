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
# Options naming the source class and its file; a class-name pointer is refused on a request scoped to a definition.
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
        name, chosen = cls.get_class_name(), options.get(ORDER_SOURCE)
        path, needs = options.get(ORDERS), options.get(NEEDS)
        reason = None
        if chosen != name:
            # Scoped here but not chosen: say which source was; a plain non-match stays silent.
            if chosen is not None and cls.is_pointed(feature_name, options, data_access_collection):
                reason = f"the request reads its orders from {chosen}"
        elif not isinstance(path, (str, Path)) or not Path(path).is_file():
            reason = f"no order file at {path}"
        elif needs is not None and needs not in cls.delivers:
            reason = f"needs {needs}; {cls.label} delivers {' and '.join(sorted(cls.delivers))} only"
        else:
            return [SourceMatch(source=str(path), access=Path(path))]
        if reason is not None:
            record_match_rejection(name, reason, stage=INPUT_DATA_STAGE)
        return []

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
