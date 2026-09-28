"""The departments' numbers. A docstring says what a number means, OWNER who answers for it."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
from mloda.provider import FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options

from mloda_demo.one_process.chain import PandasOnly
from mloda_demo.retail.sources import AS_OF, CANCELLED, NEEDS, MarketingExport, OrderSource, ShopLedger

LIMIT = 500.0  # pay later needs this net spend in the last 30 days


class Kpi(PandasOnly):
    NAMES: ClassVar[tuple[str, ...]] = ()
    OWNER: ClassVar[str] = ""
    SOURCE: ClassVar[type[OrderSource]] = ShopLedger
    NEEDS: ClassVar[str | None] = None  # what the source has to deliver

    @classmethod
    def feature_names_supported(cls) -> set[str]:
        return set(cls.NAMES)

    @classmethod
    def inputs(cls, *names: str, needs: str | None = None) -> set[Feature]:
        """Input features that carry what this definition needs from the source."""
        needs = needs or cls.NEEDS
        return {Feature(name, Options(group={} if needs is None else {NEEDS: needs})) for name in names}


def as_of(features: FeatureSet) -> pd.Timestamp:
    return pd.Timestamp(features.get_options_key(AS_OF))


def days_before(data: pd.DataFrame, value: str, end: Any, days: int) -> pd.Series:
    """The customer's sum of `value` over the `days` before `end`, on every row of that customer."""
    inside = (data["invoice_date"] >= end - pd.Timedelta(days=days)) & (data["invoice_date"] < end)
    return data[value].where(inside, 0).groupby(data["customer_id"]).transform("sum")


class LineValue(Kpi, FeatureGroup):
    """Quantity times price; a cancellation counts negative."""

    NAMES = ("line_value",)
    OWNER = "finance"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("quantity", "price")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["line_value"] = data["quantity"] * data["price"]
        return data


class GrossSpend(Kpi, FeatureGroup):
    """All orders in the 30 days before the checkout."""

    NAMES = ("gross_spend",)
    OWNER = "marketing"
    SOURCE = MarketingExport

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("line_value", "customer_id", "invoice_date")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["gross_spend"] = days_before(data, "line_value", as_of(features), days=30)
        return data


class NetSpend30d(Kpi, FeatureGroup):
    """Orders minus cancellations in the 30 days before the checkout. Decides pay later."""

    NAMES = ("net_spend_30d",)
    OWNER = "risk"
    NEEDS = "cancellations"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("line_value", "customer_id", "invoice_date")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["net_spend_30d"] = days_before(data, "line_value", as_of(features), days=30)
        return data


class PayLater(Kpi, FeatureGroup):
    """The checkout's rule: pay later needs a net spend of 500 in the last 30 days."""

    NAMES = ("pay_later",)
    OWNER = "risk"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("net_spend_30d")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["pay_later"] = data["net_spend_30d"] >= LIMIT
        return data


class LastReturn(Kpi, FeatureGroup):
    """The customer's last cancellation before the checkout: its date and its value."""

    NAMES = ("last_return", "last_return_value")
    OWNER = "logistics"
    NEEDS = "cancellations"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("invoice", "customer_id", "invoice_date", "line_value")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        customer = data["customer_id"]
        returns = data["invoice"].str.startswith(CANCELLED) & (data["invoice_date"] < as_of(features))
        data["last_return"] = data["invoice_date"].where(returns).groupby(customer).transform("max")
        last = returns & (data["invoice_date"] == data["last_return"])
        data["last_return_value"] = -data["line_value"].where(last, 0).groupby(customer).transform("sum")
        return data


class DaysBefore(Kpi, FeatureChainParserMixin, FeatureGroup):
    """`value__7d_before__event`: the customer's sum of a value in the days before an event's date."""

    PREFIX_PATTERN = r"^[a-z_]+__\d+d_before__[a-z_]+$"
    OWNER = "shared"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        value, _, event = str(feature_name).split("__")
        # The window needs what its parts need, so both parts read the same source.
        return self.inputs(value, event, "customer_id", "invoice_date", needs=needs_of(value) or needs_of(event))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            value, days, event = str(feature.name).split("__")
            data[str(feature.name)] = days_before(data, value, data[event], days=int(days.split("d_")[0]))
        return data


DEFINITIONS: tuple[type[Kpi], ...] = (LineValue, GrossSpend, NetSpend30d, PayLater, LastReturn, DaysBefore)
FEATURE_GROUPS: tuple[type[FeatureGroup], ...] = (LineValue, GrossSpend, NetSpend30d, PayLater, LastReturn, DaysBefore)


def needs_of(name: str) -> str | None:
    return next((kpi.NEEDS for kpi in DEFINITIONS if name in kpi.NAMES), None)


def kpi_of(name: str) -> type[Kpi] | None:
    return next((kpi for kpi in DEFINITIONS if name in kpi.NAMES), None)
