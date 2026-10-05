from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from mloda.provider import FeatureResolutionError
from mloda.user import Feature, Options, PluginCollector, get_feature_group_docs, mloda

from mloda_demo.retail.kpis import NetSpend30d
from mloda_demo.retail.picture import graph
from mloda_demo.retail.runs import (
    CHECKOUT,
    DEFINITION_GROUPS,
    decide,
    ensure_store,
    fill_store,
    run,
    show,
    steps,
)
from mloda_demo.retail.runs import CUSTOMER_ID as CUSTOMER
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


def before_checkout(frame: pd.DataFrame, days: int | None = None) -> pd.DataFrame:
    end = pd.Timestamp(CHECKOUT)
    start = end - pd.Timedelta(days=days) if days else frame["invoice_date"].min()
    return frame[(frame["invoice_date"] >= start) & (frame["invoice_date"] < end)]


def test_the_checkout_declines_the_customer() -> None:
    decision = decide(CUSTOMER)
    assert not decision.allowed
    assert decision.net_spend == pytest.approx(252.0) == pytest.approx(STORE.values[CUSTOMER])


def test_marketing_and_risk_differ_by_exactly_the_cancelled_order(ledger: pd.DataFrame) -> None:
    gross = run(["gross_spend"]).value("gross_spend", CUSTOMER)
    net = run(["net_spend_30d"]).value("net_spend_30d", CUSTOMER)
    window = before_checkout(ledger[ledger["customer_id"] == CUSTOMER], days=30)
    cancelled = window[window["invoice"].str.startswith("C")]["value"].sum()
    assert (gross, net) == (pytest.approx(1591.6), pytest.approx(252.0))
    assert gross - net == pytest.approx(-cancelled) == pytest.approx(1339.6)
    # Marketing's definition leaves cancellations out itself, whichever source feeds it.
    assert run(["gross_spend"], source=ShopLedger).value("gross_spend", CUSTOMER) == pytest.approx(gross)


@pytest.mark.parametrize(("name", "days"), [("net_spend_30d", 30), ("net_spend", None)])
def test_net_spend_matches_pandas_for_every_customer(ledger: pd.DataFrame, name: str, days: int | None) -> None:
    result = run([name]).frame.drop_duplicates("customer_id").set_index("customer_id")[name]
    expected = before_checkout(ledger, days).groupby("customer_id")["value"].sum()
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
    fill_store()
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


def at_checkout(source: str = "ShopLedger", **group: Any) -> Options:
    return Options(group={source: str(LEDGER), AS_OF: CHECKOUT, **group})


def plugins(groups: frozenset[Any]) -> dict[str, Any]:
    return {
        "compute_frameworks": ["PandasDataFrame"],
        "plugin_collector": PluginCollector.enabled_feature_groups(set(groups)),
    }


def test_the_registry_search_finds_the_three_spends() -> None:
    entries = [
        entry for entry in get_feature_group_docs(search="spend") if entry.module.startswith("mloda_demo.retail")
    ]
    assert [entry.name for entry in entries] == ["GrossSpend", "NetSpend", "NetSpend30d"]
    page = show(entries)._mime_()[1]
    assert "<td>net_spend_30d</td><td>risk</td><td>Net spend, 30 days: orders minus cancellations." in page
    assert (
        "<td>gross_spend</td><td>marketing</td><td>Gross spend, 30 days: all orders, cancellations ignored.</td>"
        in page
    )


def test_one_customer_reads_only_that_customers_order_lines() -> None:
    result = mloda.run_all([Feature("customer_id", at_checkout(customer=CUSTOMER))], **plugins(DEFINITION_GROUPS))
    assert set(result[0]["customer_id"]) == {CUSTOMER}


def test_one_request_scoped_per_feature_gets_one_number_from_the_definition_and_the_store() -> None:
    ensure_store()
    result = mloda.run_all(
        [
            Feature("net_spend_30d", at_checkout(), feature_group="NetSpend30d"),
            Feature(
                "net_spend_30d", Options(group={AS_OF: CHECKOUT, "customer": CUSTOMER}), feature_group="FeatureStore"
            ),
            Feature("net_spend_30d", at_checkout(customer=CUSTOMER), feature_group="NetSpend30d"),
        ]
    )
    answered = [
        (step.feature_group.__name__, len(frame), round(float(frame["net_spend_30d"].iloc[0]), 2))
        for step, frame in zip([step for step in result.plan if step.requested_feature_names], result, strict=True)
    ]
    assert ("FeatureStore", 1, 252.0) in answered
    assert any(group == "NetSpend30d" and rows > 1000 for group, rows, _ in answered), "training: every order line"
    page = show(result)._mime_()[1]
    assert "FeatureStore, customer 14045" in page and "NetSpend30d, every customer" in page
    assert "NetSpend30d, customer 14045" in page and "[85381 rows x 1 columns]" in page


def test_a_refusal_shows_its_reason() -> None:
    with pytest.raises(FeatureResolutionError) as refusal:
        mloda.run_all([Feature("net_spend_30d", at_checkout("MarketingExport"))], **plugins(DEFINITION_GROUPS))
    page = show(refusal.value)._mime_()[1]
    assert "FeatureResolutionError: No feature groups found" in page
    assert "needs cancellations; the marketing export delivers orders only" in page


def test_a_registered_number_shows_its_definition_and_a_new_one_its_plan() -> None:
    one = at_checkout(customer=CUSTOMER)
    registered = show(mloda.run_all([Feature("net_spend_30d", one)], **plugins(DEFINITION_GROUPS)))._mime_()[1]
    composed = mloda.run_all([Feature("line_value__7d_before__last_return", one)], **plugins(DEFINITION_GROUPS))
    assert "class NetSpend30d" in registered
    assert "<svg" in show(composed)._mime_()[1]


def test_the_returned_order_was_the_big_one() -> None:
    names = ["line_value__7d_before__last_return", "last_return_value"]
    result = run(names)
    assert result.value(names[0], CUSTOMER) == pytest.approx(1339.6)
    assert result.value(names[1], CUSTOMER) == pytest.approx(1339.6)


def test_the_plan_joins_finance_and_logistics_before_data_moves() -> None:
    nodes, edges = graph(steps(["line_value__7d_before__last_return"]), ShopLedger)
    window = next(index for index, node in enumerate(nodes) if node.owner == "on request")
    assert {nodes[start].owner for start, end in edges if end == window} == {"finance", "logistics"}
    assert sorted(node.owner for node in nodes) == ["finance", "finance", "logistics", "on request"]
