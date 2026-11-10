from collections.abc import Callable

import pandas as pd
import pytest
from mloda.provider import FeatureSet, SourceFeatureGroup, SourceMatch
from mloda.user import Options

from mloda_demo.retail.kpis import (
    DaysBefore,
    GrossSpend,
    LastReturn,
    NetSpend,
    NetSpend30d,
    PayLater,
    days_before,
    needs_of,
    parse,
)
from mloda_demo.retail.sources import LEDGER, NEEDS, ORDER_SOURCE, ORDERS, MarketingExport, OrderSource, ShopLedger
from mloda_demo.retail.store import STORE, FeatureStore

FeatureSetFactory = Callable[..., FeatureSet]
LEDGER_PATH = str(LEDGER)


def lines(*rows: tuple[str, int, str, float]) -> pd.DataFrame:
    """Order lines as (invoice, customer, date, value)."""
    frame = pd.DataFrame(rows, columns=["invoice", "customer_id", "invoice_date", "line_value"])
    frame["invoice_date"] = pd.to_datetime(frame["invoice_date"])
    return frame


def test_numbers_match_by_name_and_the_window_by_pattern() -> None:
    assert NetSpend30d.match_feature_group_criteria("net_spend_30d", Options())
    assert LastReturn.feature_names_supported() == {"last_return", "last_return_value"}
    assert DaysBefore.match_feature_group_criteria("line_value__7d_before__last_return", Options())
    assert DaysBefore.match_feature_group_criteria("net_spend_30d__14d_before__last_return", Options())
    assert not DaysBefore.match_feature_group_criteria("net_spend_30d", Options())
    assert parse("net_spend_30d__14d_before__last_return") == ("net_spend_30d", 14, "last_return")


def test_the_window_needs_what_its_parts_need() -> None:
    assert needs_of("last_return") == needs_of("net_spend_30d") == "cancellations"
    assert needs_of("line_value") is None
    features = DaysBefore().input_features(Options(), "line_value__7d_before__last_return")  # type: ignore[arg-type]
    assert features is not None
    assert {feature.options.get(NEEDS) for feature in features} == {"cancellations"}


def test_days_before_counts_the_start_and_leaves_out_the_end() -> None:
    data = lines(
        ("1", 7, "2010-08-01 00:00", 10.0),  # exactly 30 days before: in
        ("2", 7, "2010-08-20 00:00", 5.0),
        ("3", 7, "2010-08-31 00:00", 100.0),  # the end itself: out
        ("4", 8, "2010-08-20 00:00", 1.0),
    )
    window = days_before(data, data["line_value"], pd.Timestamp("2010-08-31"), days=30)
    assert list(window) == [15.0, 15.0, 15.0, 1.0]


def test_a_cancellation_lowers_net_spend_and_not_gross_spend(feature_set: FeatureSetFactory) -> None:
    data = lines(("1", 7, "2010-08-05", 1339.6), ("C1", 7, "2010-08-06", -1339.6), ("2", 7, "2010-08-17", 252.0))
    net = NetSpend30d.calculate_feature(data.copy(), feature_set("net_spend_30d", as_of="2010-08-31 15:37"))
    gross = GrossSpend.calculate_feature(data.copy(), feature_set("gross_spend", as_of="2010-08-31 15:37"))
    assert net["net_spend_30d"].iloc[0] == 252.0
    assert gross["gross_spend"].iloc[0] == 1591.6
    assert not PayLater.calculate_feature(net, feature_set("pay_later"))["pay_later"].iloc[0]


def test_finance_counts_every_order_before_the_checkout(feature_set: FeatureSetFactory) -> None:
    data = lines(
        ("0", 7, "2010-06-01", 40.0),  # older than 30 days: in
        ("1", 7, "2010-08-05", 1339.6),
        ("C1", 7, "2010-08-06", -1339.6),
        ("2", 7, "2010-09-02", 99.0),  # after the checkout: out
    )
    net = NetSpend.calculate_feature(data, feature_set("net_spend", as_of="2010-08-31 15:37"))
    assert list(net["net_spend"]) == pytest.approx([40.0] * 4)


def test_last_return_ignores_returns_after_the_checkout(feature_set: FeatureSetFactory) -> None:
    data = lines(
        ("1", 7, "2010-08-05", 1339.6),
        ("C1", 7, "2010-08-06", -1339.6),
        ("C2", 7, "2010-09-02", -40.8),  # after the checkout
        ("2", 8, "2010-08-17", 252.0),  # no returns at all
    )
    result = LastReturn.calculate_feature(data, feature_set("last_return", as_of="2010-08-31 15:37"))
    assert result["last_return"].iloc[0] == pd.Timestamp("2010-08-06")
    assert result["last_return_value"].iloc[0] == 1339.6
    assert pd.isna(result["last_return"].iloc[3])
    assert str(result["last_return_value"].iloc[3]) == "0.0"  # not -0.0


def test_last_return_is_one_invoice_even_when_two_share_a_minute(feature_set: FeatureSetFactory) -> None:
    data = lines(("C1", 7, "2010-08-06 11:06", -10.0), ("C2", 7, "2010-08-06 11:06", -20.0))
    result = LastReturn.calculate_feature(data, feature_set("last_return_value", as_of="2010-08-31 15:37"))
    assert result["last_return_value"].iloc[0] == 20.0


def test_window_before_an_event_is_parsed_from_the_name(feature_set: FeatureSetFactory) -> None:
    data = lines(("1", 7, "2010-08-05", 1339.6), ("C1", 7, "2010-08-06", -1339.6), ("2", 8, "2010-08-05", 5.0))
    data["last_return"] = [pd.Timestamp("2010-08-06"), pd.Timestamp("2010-08-06"), pd.NaT]  # customer 8: no return
    name = "line_value__7d_before__last_return"
    assert list(DaysBefore.calculate_feature(data, feature_set(name))[name]) == [1339.6, 1339.6, 0.0]


def test_the_store_answers_from_what_the_batch_stored(feature_set: FeatureSetFactory) -> None:
    STORE.fill({7: 252.0}, "v", "t")
    try:
        result = FeatureStore.calculate_feature(None, feature_set("net_spend_30d", customer=7))
    finally:
        STORE.clear()
    assert list(result["net_spend_30d"]) == [252.0]


def claims(source: type[OrderSource], name: str, path: str = LEDGER_PATH, **group: str) -> bool:
    return source.match_feature_group_criteria(
        name, Options(group={ORDER_SOURCE: source.__name__, ORDERS: path, **group})
    )


def test_both_sources_answer_to_the_family_scope() -> None:
    assert issubclass(OrderSource, SourceFeatureGroup)
    assert "OrderSource" in ShopLedger.pointer_keys() and "OrderSource" in MarketingExport.pointer_keys()


def test_sources_claim_order_columns_only() -> None:
    assert claims(ShopLedger, "invoice") and claims(ShopLedger, "price")
    assert not claims(ShopLedger, "line_value")
    assert not claims(ShopLedger, "invoice", "missing.csv.gz")


def test_marketing_export_declines_a_definition_that_needs_cancellations() -> None:
    assert not claims(MarketingExport, "invoice", needs="cancellations")
    assert claims(ShopLedger, "invoice", needs="cancellations")


def test_marketing_export_leaves_out_cancellations(feature_set: FeatureSetFactory) -> None:
    match = SourceMatch(LEDGER_PATH, LEDGER_PATH)
    ledger = ShopLedger.load_neutral(match, feature_set("invoice"))
    export = MarketingExport.load_neutral(match, feature_set("invoice"))
    cancelled = ledger["invoice"].str.startswith("C")
    assert cancelled.any()
    assert len(export) == (~cancelled).sum()
    assert not export["invoice"].str.startswith("C").any()
