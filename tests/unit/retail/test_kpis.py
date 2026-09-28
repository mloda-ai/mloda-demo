from collections.abc import Callable

import pandas as pd
from mloda.provider import FeatureSet
from mloda.user import Options

from mloda_demo.retail.kpis import DaysBefore, LastReturn, NetSpend30d, PayLater, days_before, needs_of
from mloda_demo.retail.sources import LEDGER, NEEDS, MarketingExport, Orders, OrderSource, ShopLedger

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
    assert not DaysBefore.match_feature_group_criteria("net_spend_30d", Options())


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
    assert list(days_before(data, "line_value", pd.Timestamp("2010-08-31"), days=30)) == [15.0, 15.0, 15.0, 1.0]


def test_a_cancellation_lowers_net_spend(feature_set: FeatureSetFactory) -> None:
    data = lines(("1", 7, "2010-08-05", 1339.6), ("C1", 7, "2010-08-06", -1339.6), ("2", 7, "2010-08-17", 252.0))
    result = NetSpend30d.calculate_feature(data, feature_set("net_spend_30d", as_of="2010-08-31 15:37"))
    assert result["net_spend_30d"].iloc[0] == 252.0
    assert not PayLater.calculate_feature(result, feature_set("pay_later"))["pay_later"].iloc[0]


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
    assert result["last_return_value"].iloc[3] == 0


def test_window_before_an_event_is_parsed_from_the_name(feature_set: FeatureSetFactory) -> None:
    data = lines(("1", 7, "2010-08-05", 1339.6), ("C1", 7, "2010-08-06", -1339.6))
    data["last_return"] = pd.Timestamp("2010-08-06")
    name = "line_value__7d_before__last_return"
    assert list(DaysBefore.calculate_feature(data, feature_set(name))[name]) == [1339.6, 1339.6]


def test_root_feature_group_reads_through_the_source_family() -> None:
    assert isinstance(Orders.input_data(), OrderSource)


def test_sources_claim_order_columns_only() -> None:
    assert ShopLedger.match_subclass_data_access(LEDGER_PATH, ["invoice", "price"], Options()) == LEDGER_PATH
    assert ShopLedger.match_subclass_data_access(LEDGER_PATH, ["line_value"], Options()) is None
    assert ShopLedger.match_subclass_data_access("missing.csv.gz", ["invoice"], Options()) is None


def test_marketing_export_declines_a_definition_that_needs_cancellations() -> None:
    options = Options(group={NEEDS: "cancellations"})
    assert MarketingExport.match_subclass_data_access(LEDGER_PATH, ["invoice"], options) is None
    assert ShopLedger.match_subclass_data_access(LEDGER_PATH, ["invoice"], options) == LEDGER_PATH


def test_marketing_export_leaves_out_cancellations(feature_set: FeatureSetFactory) -> None:
    ledger = ShopLedger.load_data(LEDGER_PATH, feature_set("invoice"))
    export = MarketingExport.load_data(LEDGER_PATH, feature_set("invoice"))
    cancelled = ledger["invoice"].str.startswith("C")
    assert cancelled.any()
    assert len(export) == (~cancelled).sum()
    assert not export["invoice"].str.startswith("C").any()
