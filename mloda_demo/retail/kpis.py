"""The departments' numbers. A docstring says what a number means, OWNER who answers for it."""

from __future__ import annotations

import re
from typing import Any, ClassVar, cast

import pandas as pd
from mloda.provider import FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import DataType, Feature, FeatureName, Options

from mloda_demo.pandas_only import PandasOnly
from mloda_demo.retail.sources import AS_OF, CANCELLED, COLUMNS, NEEDS, MarketingExport, OrderSource, ShopLedger

LIMIT = 500.0  # pay later needs this net spend in the last 30 days
WINDOW = re.compile(r"^(?P<value>[a-z0-9_]+?)__(?P<days>\d+)d_before__(?P<event>[a-z0-9_]+)$")


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
        return {
            Feature(
                name,
                Options(group={} if needs is None else {NEEDS: needs}),
                feature_group="Orders" if name in COLUMNS else None,  # raw columns come from the shop's orders
            )
            for name in names
        }


def as_of(features: FeatureSet) -> pd.Timestamp:
    return pd.Timestamp(features.get_options_key(AS_OF))


def days_before(data: pd.DataFrame, values: pd.Series, end: Any, days: int) -> pd.Series:
    """The customer's sum of `values` over the `days` before `end`, on every row of that customer."""
    inside = (data["invoice_date"] >= end - pd.Timedelta(days=days)) & (data["invoice_date"] < end)
    return values.where(inside, 0).groupby(data["customer_id"]).transform("sum")


def cancelled(data: pd.DataFrame) -> pd.Series:
    return data["invoice"].str.startswith(CANCELLED)


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
    """Gross spend, 30 days: all orders, cancellations ignored."""

    NAMES = ("gross_spend",)
    OWNER = "marketing"
    SOURCE = MarketingExport

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("line_value", "invoice", "customer_id", "invoice_date")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        orders = data["line_value"].where(~cancelled(data), 0)
        data["gross_spend"] = days_before(data, orders, as_of(features), days=30)
        return data


class NetSpend(Kpi, FeatureGroup):
    """Net spend: orders minus cancellations, all time."""

    NAMES = ("net_spend",)
    OWNER = "finance"
    NEEDS = "cancellations"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return self.inputs("line_value", "customer_id", "invoice_date")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        before = data["invoice_date"] < as_of(features)
        data["net_spend"] = data["line_value"].where(before, 0).groupby(data["customer_id"]).transform("sum")
        return data


class NetSpend30d(Kpi, FeatureGroup):
    """Net spend, 30 days: orders minus cancellations. The checkout uses it."""

    NAMES = ("net_spend_30d",)
    OWNER = "risk"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        return {
            Feature("line_value", options={"needs": "cancellations"}),
            Feature("customer_id", options={"needs": "cancellations"}, feature_group="Orders"),
            Feature("invoice_date", options={"needs": "cancellations"}, feature_group="Orders"),
        }

    @classmethod
    def return_data_type_rule(cls, feature: Feature) -> DataType | None:
        return DataType.DOUBLE

    @classmethod
    def validate_output_features(cls, data: Any, features: FeatureSet) -> None:
        if data["net_spend_30d"].isna().any():
            raise ValueError("net_spend_30d: a customer has no value")

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        data["net_spend_30d"] = days_before(data, data["line_value"], as_of(features), days=30)
        return data


class PayLater(Kpi, FeatureGroup):
    """The checkout's rule: pay by invoice needs 500 of orders minus cancellations in the last 30 days."""

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
        returns = cancelled(data) & (data["invoice_date"] < as_of(features))
        data["last_return"] = data["invoice_date"].where(returns).groupby(customer).transform("max")
        latest = data[returns & (data["invoice_date"] == data["last_return"])]
        last_invoice = customer.map(latest.groupby("customer_id")["invoice"].max())
        value = data["line_value"].where(data["invoice"] == last_invoice, 0).groupby(customer).transform("sum")
        data["last_return_value"] = -value + 0.0  # + 0.0 turns -0.0 into 0.0
        return data


class DaysBefore(Kpi, FeatureChainParserMixin, FeatureGroup):
    """`value__7d_before__event`: the customer's sum of a value in the days before an event's date."""

    PREFIX_PATTERN = WINDOW.pattern
    OWNER = "on request"  # nobody registers a combination; it is resolved when asked for

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        value, _, event = parse(feature_name)
        # The window needs what its parts need, so both parts read the same source.
        return self.inputs(value, event, "customer_id", "invoice_date", needs=needs_of(value) or needs_of(event))

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            value, days, event = parse(feature.name)
            data[str(feature.name)] = days_before(data, data[value], data[event], days=days)
        return data


def parse(name: FeatureName | str) -> tuple[str, int, str]:
    match = WINDOW.match(str(name))
    if match is None:
        raise ValueError(f"{name} is not value__<n>d_before__event")
    return match["value"], int(match["days"]), match["event"]


DEFINITIONS: tuple[type[Kpi], ...] = (LineValue, GrossSpend, NetSpend, NetSpend30d, PayLater, LastReturn, DaysBefore)
FEATURE_GROUPS = tuple(cast(type[FeatureGroup], kpi) for kpi in DEFINITIONS)


def needs_of(name: str) -> str | None:
    """What a registered number's inputs ask of the source, read from its input_features."""
    kpi = kpi_of(name)
    inputs = cast(type[FeatureGroup], kpi)().input_features(Options(), FeatureName(name)) if kpi else None
    return next((needs for feature in inputs or () if (needs := feature.options.get(NEEDS))), None)


def kpi_of(name: str) -> type[Kpi] | None:
    return next((kpi for kpi in DEFINITIONS if name in kpi.NAMES), None)
