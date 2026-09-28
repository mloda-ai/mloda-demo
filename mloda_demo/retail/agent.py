"""The support agent: its questions are scripted, its calls are real mloda runs. Every answer renders as a slide."""

from __future__ import annotations

import html
import inspect
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from mloda.provider import FeatureGroup, FeatureResolutionError
from mloda.user import Feature, Options, PluginCollector, get_feature_group_docs, mloda

from mloda_demo.retail.kpis import FEATURE_GROUPS, LIMIT, NetSpend30d, PayLater, kpi_of
from mloda_demo.retail.picture import OWNER_COLOURS, svg
from mloda_demo.retail.sources import AS_OF, LEDGER, MarketingExport, Orders, OrderSource, ShopLedger
from mloda_demo.retail.store import CUSTOMER as STORE_CUSTOMER
from mloda_demo.retail.store import STORE, FeatureStore

CHECKOUT = "2010-08-31 15:37"
CUSTOMER = 14045
# The definition and the store both serve net_spend_30d; a run enables one of them, never both.
DEFINITION_GROUPS: frozenset[type[FeatureGroup]] = frozenset({Orders, *FEATURE_GROUPS})
STORE_GROUPS: frozenset[type[FeatureGroup]] = frozenset({FeatureStore, PayLater})


@dataclass(frozen=True)
class Run:
    frame: pd.DataFrame
    seconds: float
    refusal: str | None = None

    def value(self, name: str, customer: int) -> Any:
        rows = self.frame[self.frame["customer_id"] == customer] if "customer_id" in self.frame else self.frame
        return rows[name].iloc[0]


def run(names: Sequence[str], *, source: type[OrderSource] | None = None, customer: int | None = None) -> Run:
    """One mloda run at the checkout. With `customer`, the store answers; otherwise the definitions do."""
    if customer is not None:
        group: dict[Any, Any] = {AS_OF: CHECKOUT, STORE_CUSTOMER: customer}
        groups, requested = STORE_GROUPS, list(names)
    else:
        kpi = kpi_of(names[0])
        group = {source or (kpi.SOURCE if kpi else ShopLedger): str(LEDGER), AS_OF: CHECKOUT}
        groups, requested = DEFINITION_GROUPS, [*names, "customer_id"]
    start = time.perf_counter()
    try:
        frames = mloda.run_all(
            [Feature(name, Options(group=dict(group))) for name in requested],
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(set(groups)),
        )
    except FeatureResolutionError as error:
        reasons = "; ".join(elimination.reason for elimination in error.result.eliminations.values())
        kpi = kpi_of(names[0])
        consumer = f"{names[0]} ({kpi.OWNER})" if kpi else names[0]
        return Run(pd.DataFrame(), time.perf_counter() - start, f"{consumer} {reasons or error}")
    took = time.perf_counter() - start
    # Every frame comes from the same order lines in the same order, so they line up.
    if any(not frame.index.equals(frames[0].index) for frame in frames):
        raise ValueError("result frames are not row-aligned")
    return Run(pd.concat(list(frames), axis=1), took)


def plan(names: Sequence[str], source: type[OrderSource] = ShopLedger) -> str:
    """The plan picture, from mloda's resolved plan; nothing is computed."""
    steps = mloda.explain(
        [Feature(name, Options(group={source.__name__: str(LEDGER), AS_OF: CHECKOUT})) for name in names],
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups(set(DEFINITION_GROUPS)),
    )
    return svg(steps, source)


def batch() -> Run:
    """Every customer's net_spend_30d at the checkout, into the store."""
    result = run(["net_spend_30d"])
    per_customer = result.frame.drop_duplicates("customer_id")
    values = dict(zip(per_customer["customer_id"].astype(int), per_customer["net_spend_30d"].astype(float)))
    STORE.fill(values, NetSpend30d.version(), CHECKOUT)
    return result


def ensure_store() -> None:
    if not STORE.values:
        batch()


def registry(word: str) -> list[dict[str, str]]:
    """What exists, from the installed package: one row per number whose name contains `word`."""
    docs = get_feature_group_docs(plugin_collector=PluginCollector.enabled_feature_groups(set(DEFINITION_GROUPS)))
    rows = []
    for doc in docs:
        for name in sorted(doc.supported_feature_names):
            kpi = kpi_of(name)
            if word in name and kpi is not None:
                rows.append(
                    {
                        "name": name,
                        "owner": kpi.OWNER,
                        "meaning": doc.description,
                        "needs": kpi.NEEDS or "orders",
                        "version": short(doc.version),
                    }
                )
    return rows


def short(version: str) -> str:
    """The code hash at the end of a feature group version."""
    return version.rsplit("-", 1)[-1][:8]


# Rendering: every call returns an object marimo shows as one slide.


def seconds(value: float) -> str:
    return f'<span class="retail-time">{value * 1000:,.0f} ms</span>'


def owner_tag(owner: str) -> str:
    colour = OWNER_COLOURS.get(owner, "#6A6C6A")
    return f'<span class="retail-owner" style="background:{colour}">{html.escape(owner)}</span>'


def money(value: float) -> str:
    return f"{value:,.2f}"


@dataclass(frozen=True)
class Slide:
    body: str
    title: str = ""
    step: str = ""

    def _mime_(self) -> tuple[str, str]:
        step = f'<p class="retail-step">{html.escape(self.step)}</p>' if self.step else ""
        title = f"<h2>{html.escape(self.title)}</h2>" if self.title else ""
        return "text/html", f'<div class="retail">{step}{title}{self.body}</div>'


def receipt(title: str, value: str, lines: Sequence[str], took: float) -> str:
    rows = "".join(f"<li>{line}</li>" for line in lines)
    return (
        f'<div class="retail-receipt"><h3>{html.escape(title)}</h3><p class="retail-value">{value}</p>'
        f"<ul>{rows}</ul>{seconds(took)}</div>"
    )


def table(rows: Sequence[dict[str, str]], owner_column: str = "owner") -> str:
    head = "".join(f"<th>{html.escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{owner_tag(value) if key == owner_column else html.escape(value)}</td>" for key, value in row.items()
        )
        + "</tr>"
        for row in rows
    )
    return f'<table class="retail-table"><tr>{head}</tr>{body}</table>'


def org() -> Slide:
    rows = [
        {"department": "finance", "numbers": "the shop ledger, line_value"},
        {"department": "marketing", "numbers": "the marketing export, gross_spend"},
        {"department": "risk", "numbers": "net_spend_30d, pay_later at the checkout"},
        {"department": "logistics", "numbers": "last_return"},
        {"department": "support", "numbers": "runs the AI agent"},
    ]
    credit = (
        '<p class="retail-credit">The company is simulated. The data is real: UCI Online Retail II '
        "(CC BY 4.0), a UK online gift shop, June to August 2010.</p>"
    )
    return Slide(table(rows, owner_column="department") + credit, "A small company")


def checkout(customer: int = CUSTOMER) -> Slide:
    """The live system: pay later for one customer, from the store."""
    ensure_store()
    result = run(["pay_later", "net_spend_30d"], customer=customer)
    allowed = bool(result.value("pay_later", customer))
    verdict = "Pay later: accepted" if allowed else "Pay later: declined"
    value = money(result.value("net_spend_30d", customer))
    body = receipt(
        f"Checkout, customer {customer}, {CHECKOUT}",
        verdict,
        [f"net_spend_30d = {value} {owner_tag('risk')}", f"rule: at least {LIMIT:,.0f}", "read from the store"],
        result.seconds,
    )
    return Slide(body, "The checkout says no")


@dataclass
class Agent:
    """Asks mloda what it needs; each question maps to the calls it makes."""

    customer: int = CUSTOMER
    script: dict[str, Callable[[Agent], Slide]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.script = {
            "Why was I declined?": Agent.why_declined,
            "Which numbers are called spend?": Agent.which_spend,
            "What does net_spend_30d mean?": Agent.meaning,
            "Can I compute net_spend_30d from the marketing export?": Agent.from_marketing,
            "Which number did the checkout use?": Agent.checkout_number,
            "Was the returned order the big one?": Agent.big_return,
        }

    def ask(self, question: str) -> Slide:
        return self.script[question](self)

    def say(self, question: str, calls: Sequence[str], answer: str, extra: str = "", step: str = "") -> Slide:
        code = "".join(f"<code>{html.escape(call)}</code>" for call in calls)
        block = f'<pre class="call">{code}</pre>' if calls else ""
        body = (
            f'<p class="retail-question">{html.escape(question)}</p>{block}{extra}<p class="retail-answer">{answer}</p>'
        )
        return Slide(body, step=step or "The support agent")

    def why_declined(self) -> Slide:
        name = registry("spend")[0]["name"]  # the first number called spend
        result = run([name])
        value = result.value(name, self.customer)
        answer = f"Your spend is {money(value)}, above the {LIMIT:,.0f} limit. You should have been accepted."
        calls = ['registry("spend")[0]', f'mloda.run_all(["{name}"])']
        return self.say("Why was I declined?", calls, answer + " " + seconds(result.seconds))

    def which_spend(self) -> Slide:
        rows = registry("spend")
        answer = "Two numbers called spend, two owners. The checkout decides with the one risk owns."
        return self.say(
            "Which numbers are called spend?", ['registry("spend")'], answer, table(rows), "1 · Find what exists"
        )

    def meaning(self) -> Slide:
        source = "\n".join(line for line in inspect.getsource(NetSpend30d).splitlines() if line.strip())
        code = f'<pre class="retail-code"><code>{html.escape(source)}</code></pre>'
        result = run(["net_spend_30d"])
        value = money(result.value("net_spend_30d", self.customer))
        answer = f"Orders minus cancellations, 30 days before the checkout: {value}. {seconds(result.seconds)}"
        return self.say("What does net_spend_30d mean?", [], answer, code, "2 · Understand a number")

    def from_marketing(self) -> Slide:
        result = run(["net_spend_30d"], source=MarketingExport)
        refusal = f'<p class="retail-refusal">{html.escape(result.refusal or "")}</p>'
        answer = f"No. mloda stopped while planning, before any data was read. {seconds(result.seconds)}"
        calls = ['mloda.run_all(["net_spend_30d"], MarketingExport)']
        return self.say(
            "Can I compute net_spend_30d from the marketing export?", calls, answer, refusal, "3 · Trust it"
        )

    def checkout_number(self) -> Slide:
        stored = batch()
        live = run(["pay_later", "net_spend_30d"], customer=self.customer)
        mine = run(["net_spend_30d"])
        stored_value = money(STORE.values[self.customer])
        value = money(mine.value("net_spend_30d", self.customer))
        version = short(STORE.version)
        receipts = (
            '<div class="retail-receipts">'
            + receipt(
                "Batch: all customers", f"{len(STORE.values):,} rows", ["into the store", version], stored.seconds
            )
            + receipt("Checkout: from the store", stored_value, ["net_spend_30d", version], live.seconds)
            + receipt("Agent: from the definition", value, ["net_spend_30d", version], mine.seconds)
            + "</div>"
        )
        answer = f"The checkout used net_spend_30d: {value}. The cancelled order does not count, so it is below 500."
        calls = ['mloda.run_all(["net_spend_30d"])  # the batch', 'mloda.run_all(["pay_later"])  # the checkout']
        return self.say(
            "Which number did the checkout use?", calls, answer, receipts, "4 · Use it: offline, online, agent"
        )

    def big_return(self) -> Slide:
        names = ["line_value__7d_before__last_return", "last_return_value"]
        picture = f'<figure class="retail-plan">{plan(names)}</figure>'
        result = run(names)
        before = money(result.value(names[0], self.customer))
        returned = money(result.value(names[1], self.customer))
        answer = (
            f"Yes. {before} in the week before the return, and the return was worth {returned}. "
            f"{seconds(result.seconds)}"
        )
        calls = ["mloda.explain([...])  # the plan, before data moves", "mloda.run_all([...])"]
        extra = f'<p class="retail-names">{" · ".join(names)}</p>{picture}'
        return self.say("Was the returned order the big one?", calls, answer, extra, "5 · Ask something new")
