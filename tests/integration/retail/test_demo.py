from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from mloda.provider import FeatureResolutionError
from mloda.user import Feature, Options, PluginCollector, mloda

from mloda_demo.retail.agent import (
    CHECKOUT,
    CUSTOMER,
    DEFINITION_GROUPS,
    batch,
    checkout,
    ensure_store,
    registry,
    run,
    steps,
)
from mloda_demo.retail.kpis import NetSpend30d
from mloda_demo.retail.picture import graph
from mloda_demo.retail.sources import AS_OF, LEDGER, MarketingExport, OrderSource, ShopLedger
from mloda_demo.retail.store import STORE, FeatureStore


@pytest.fixture(autouse=True)
def empty_store() -> Iterator[None]:
    STORE.clear()
    yield
    STORE.clear()


@pytest.fixture(scope="module")
def ledger() -> pd.DataFrame:
    frame = pd.read_csv(LEDGER, dtype={"invoice": str}, parse_dates=["invoice_date"])
    frame["value"] = frame["quantity"] * frame["price"]
    return frame


def last_30_days(frame: pd.DataFrame) -> pd.DataFrame:
    end = pd.Timestamp(CHECKOUT)
    return frame[(frame["invoice_date"] >= end - pd.Timedelta(days=30)) & (frame["invoice_date"] < end)]


def test_the_checkout_declines_the_customer() -> None:
    assert "Pay later: declined" in checkout(CUSTOMER)._mime_()[1]
    assert STORE.values[CUSTOMER] == pytest.approx(252.0)


def test_marketing_and_risk_differ_by_exactly_the_cancelled_order(ledger: pd.DataFrame) -> None:
    gross = run(["gross_spend"]).value("gross_spend", CUSTOMER)
    net = run(["net_spend_30d"]).value("net_spend_30d", CUSTOMER)
    window = last_30_days(ledger[ledger["customer_id"] == CUSTOMER])
    cancelled = window[window["invoice"].str.startswith("C")]["value"].sum()
    assert (gross, net) == (pytest.approx(1591.6), pytest.approx(252.0))
    assert gross - net == pytest.approx(-cancelled) == pytest.approx(1339.6)
    # Marketing's definition leaves cancellations out itself, whichever source feeds it.
    assert run(["gross_spend"], source=ShopLedger).value("gross_spend", CUSTOMER) == pytest.approx(gross)


def test_net_spend_30d_matches_pandas_for_every_customer(ledger: pd.DataFrame) -> None:
    result = run(["net_spend_30d"]).frame.drop_duplicates("customer_id").set_index("customer_id")["net_spend_30d"]
    expected = last_30_days(ledger).groupby("customer_id")["value"].sum()
    pd.testing.assert_series_equal(
        result.loc[expected.index], expected, check_names=False, check_index_type=False, check_exact=False
    )
    assert (result.drop(expected.index) == 0).all()


def test_marketing_export_is_refused_before_any_data_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    loads: list[Any] = []
    monkeypatch.setattr(OrderSource, "load_data", classmethod(lambda cls, *args: loads.append(cls)))
    result = run(["net_spend_30d"], source=MarketingExport)
    assert result.refusal == "net_spend_30d (risk) needs cancellations; the marketing export delivers orders only"
    assert loads == []


def test_store_and_definition_give_the_same_number() -> None:
    batch()
    stored = run(["net_spend_30d"], customer=CUSTOMER).value("net_spend_30d", CUSTOMER)
    computed = run(["net_spend_30d"]).value("net_spend_30d", CUSTOMER)
    assert stored == pytest.approx(computed)


def test_a_store_filled_by_another_definition_is_filled_again() -> None:
    STORE.fill({CUSTOMER: 0.0}, "an older definition", CHECKOUT)
    ensure_store()
    assert STORE.version == NetSpend30d.version()
    assert STORE.values[CUSTOMER] == pytest.approx(252.0)


def test_mloda_refuses_to_guess_between_store_and_definition() -> None:
    with pytest.raises(FeatureResolutionError, match="Multiple feature groups"):
        mloda.run_all(
            [Feature("net_spend_30d", Options(group={ShopLedger: str(LEDGER), AS_OF: CHECKOUT}))],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups({*DEFINITION_GROUPS, FeatureStore}),
        )


def test_the_first_number_called_spend_is_marketings() -> None:
    assert [row["name"] for row in registry("spend")] == ["gross_spend", "net_spend_30d"]


def test_the_returned_order_was_the_big_one() -> None:
    names = ["line_value__7d_before__last_return", "last_return_value"]
    result = run(names)
    assert result.value(names[0], CUSTOMER) == pytest.approx(1339.6)
    assert result.value(names[1], CUSTOMER) == pytest.approx(1339.6)


def test_the_plan_joins_finance_and_logistics_before_data_moves() -> None:
    nodes, edges = graph(steps(["line_value__7d_before__last_return", "last_return_value"]), ShopLedger)
    window = next(index for index, node in enumerate(nodes) if node.owner == "shared")
    assert {nodes[start].owner for start, end in edges if end == window} == {"finance", "logistics"}
    assert sorted(node.owner for node in nodes) == ["finance", "finance", "logistics", "shared"]
