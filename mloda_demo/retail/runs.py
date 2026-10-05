"""mloda runs at the checkout, and `show`, which formats what mloda returned for the slides."""

from __future__ import annotations

import html
import importlib
import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mloda.provider import FeatureGroup, FeatureResolutionError
from mloda.user import Feature, Options, PlanStep, PluginCollector, RunResult, mloda

from mloda_demo.retail.kpis import FEATURE_GROUPS, Kpi, NetSpend30d, PayLater, kpi_of, needs_of
from mloda_demo.retail.picture import svg
from mloda_demo.retail.sources import AS_OF, CUSTOMER, LEDGER, NEEDS, MarketingExport, Orders, OrderSource, ShopLedger
from mloda_demo.retail.store import STORE, FeatureStore

CHECKOUT = "2010-08-31 15:37"
CUSTOMER_ID = 14045
# The definition and the store both serve net_spend_30d; a run enables one of them, never both.
DEFINITION_GROUPS: frozenset[type[FeatureGroup]] = frozenset({Orders, *FEATURE_GROUPS})
STORE_GROUPS: frozenset[type[FeatureGroup]] = frozenset({FeatureStore, PayLater})


@dataclass(frozen=True)
class Run:
    frame: pd.DataFrame
    refusal: str | None = None

    def value(self, name: str, customer: int) -> Any:
        rows = self.frame[self.frame["customer_id"] == customer] if "customer_id" in self.frame else self.frame
        return rows[name].iloc[0]


def source_of(names: Sequence[str], source: type[OrderSource] | None = None) -> type[OrderSource]:
    """The given source, else the first number's own source."""
    kpi = kpi_of(names[0])
    return source or (kpi.SOURCE if kpi else ShopLedger)


def request(names: Sequence[str], source: type[OrderSource] | None = None) -> list[Feature | str]:
    """The features at the checkout; one group for all of them, so they share one read of the source."""
    group: dict[str, Any] = {source_of(names, source).__name__: str(LEDGER), AS_OF: CHECKOUT}
    needs = next((needs for needs in map(needs_of, names) if needs), None)
    if needs is not None:
        group[NEEDS] = needs
    return [Feature(name, Options(group=dict(group))) for name in names]


def run(names: Sequence[str], *, source: type[OrderSource] | None = None, customer: int | None = None) -> Run:
    """One mloda run at the checkout. With `customer`, the store answers; otherwise the definitions do."""
    if customer is not None:
        group = {AS_OF: CHECKOUT, CUSTOMER: customer}
        features: list[Feature | str] = [Feature(name, Options(group=dict(group))) for name in names]
        groups = STORE_GROUPS
    else:
        groups, features = DEFINITION_GROUPS, request([*names, "customer_id"], source)
    try:
        frames = mloda.run_all(
            features,
            compute_frameworks=["PandasDataFrame"],
            plugin_collector=PluginCollector.enabled_feature_groups(set(groups)),
        )
    except FeatureResolutionError as error:
        kpi = kpi_of(names[0])
        consumer = f"{names[0]} ({kpi.OWNER})" if kpi else names[0]
        return Run(pd.DataFrame(), f"{consumer} {reasons(error) or error}")
    # Every frame comes from the same order lines in the same order, so they line up.
    if any(not frame.index.equals(frames[0].index) for frame in frames):
        raise ValueError("result frames are not row-aligned")
    return Run(pd.concat(list(frames), axis=1))


def reasons(error: FeatureResolutionError) -> str:
    return "; ".join(elimination.reason for elimination in error.result.eliminations.values())


def steps(names: Sequence[str]) -> list[PlanStep]:
    """mloda's resolved plan; nothing is computed."""
    return mloda.explain(
        request(names),
        compute_frameworks=["PandasDataFrame"],
        plugin_collector=PluginCollector.enabled_feature_groups(set(DEFINITION_GROUPS)),
    )


def fill_store() -> Run:
    """Every customer's net_spend_30d at the checkout, into the store."""
    result = run(["net_spend_30d"])
    per_customer = result.frame.drop_duplicates("customer_id")
    values = dict(zip(per_customer["customer_id"].astype(int), per_customer["net_spend_30d"].astype(float)))
    STORE.fill(values, NetSpend30d.version(), CHECKOUT)
    return result


def ensure_store() -> None:
    """Fill the store unless it already holds this definition at this checkout."""
    if not STORE.values or STORE.version != NetSpend30d.version() or STORE.as_of != CHECKOUT:
        fill_store()


@dataclass(frozen=True)
class Decision:
    allowed: bool
    net_spend: float


def decide(customer: int = CUSTOMER_ID) -> Decision:
    """The live system: pay by invoice for one customer, decided on the stored number."""
    ensure_store()
    result = run(["pay_later", "net_spend_30d"], customer=customer)
    return Decision(bool(result.value("pay_later", customer)), float(result.value("net_spend_30d", customer)))


PLUMBING = ("NAMES", "OWNER", "@classmethod", "#")
HOOK = re.compile(r"^(?P<indent>\s*)def (?P<name>\w+)\(.*$")
WITH_BODY = {"input_features"}  # the inputs stay visible: a refusal names one of them


def definition(kpi: type[Kpi]) -> tuple[str, str]:
    """The class as written in kpis.py, split at calculate_feature; above it, each hook as its name only."""
    lines = [
        line for line in inspect.getsource(kpi).splitlines() if line.strip() and not line.strip().startswith(PLUMBING)
    ]
    split = next(i for i, line in enumerate(lines) if line.strip().startswith("def calculate_feature"))
    body = len(lines[split]) - len(lines[split].lstrip()) + 4  # deeper than a method's own line
    shared: list[str] = []
    keep_body = False
    for line in lines[:split]:
        hook = HOOK.match(line)
        if hook:
            keep_body = hook["name"] in WITH_BODY
            shared.append(f"{hook['indent']}def {hook['name']}(...)")
        elif keep_body or len(line) - len(line.lstrip()) < body:
            shared.append(line)
    return "\n".join(shared), "\n".join(lines[split:])


# `show` formats what mloda returned: plain HTML, black on white. Nothing is computed here.


@dataclass(frozen=True)
class Shown:
    body: str

    def _mime_(self) -> tuple[str, str]:
        return "text/html", f'<div class="retail">{self.body}</div>'


def show(result: Any) -> Shown:
    """Registry entries as a table, a run as what ran and its value, several requests as their frames, a refusal as mloda's error."""
    if isinstance(result, FeatureResolutionError):
        error = html.escape(f"{type(result).__name__}: {result}")
        return Shown(f'<pre class="retail-refusal"><code>{error}</code></pre>')
    if isinstance(result, RunResult):
        several = sum(1 for step in result.plan if step.requested_feature_names) > 1
        return Shown(frames(result) if several else answer(result))
    return Shown(registry(result))


def table(rows: Sequence[dict[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(key)}</th>" for key in rows[0])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(value)}</td>" for value in row.values()) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def registry(entries: Sequence[Any]) -> str:
    """`get_feature_group_docs` entries; the owner is this demo's class attribute."""
    rows = []
    for entry in entries:
        group = getattr(importlib.import_module(entry.module), entry.name)
        for name in sorted(entry.supported_feature_names):
            rows.append({"feature": name, "owner": group.OWNER, "meaning": entry.description})
    return table(rows)


def requested(result: RunResult) -> tuple[PlanStep, str]:
    step = next(step for step in result.plan if step.requested_feature_names)
    return step, step.requested_feature_names[0]


def option(step: PlanStep, key: str) -> Any:
    return step.feature_set_options.get(key) if step.feature_set_options is not None else None


def value(result: RunResult) -> str:
    """One customer's value, or how many order lines came back."""
    step, name = requested(result)
    values = result[0][name].unique()
    customer = option(step, CUSTOMER)
    if customer is not None and len(values) == 1:
        return f"customer {customer}: {values[0]:,.2f}"
    return f"every customer: {len(result[0]):,} order lines"


def answer(result: RunResult) -> str:
    """The value first; then a registered number's definition, or for a name nobody registered, the plan mloda resolved."""
    step, name = requested(result)
    kpi = kpi_of(name)
    group: object = step.feature_group
    if kpi is not None and group is kpi:
        shared, per_use_case = definition(kpi)
        shown = (
            f'<p class="retail-label">Shared freely</p><pre><code>{html.escape(shared)}</code></pre>'
            f'<p class="retail-label">Partly shared, depends on the use case</p><pre><code>{html.escape(per_use_case)}</code></pre>'
        )
    else:
        source = MarketingExport if option(step, MarketingExport.__name__) else ShopLedger
        shown = f'<figure class="retail-plan">{svg(result.plan, source)}</figure>'
    return f'<p class="retail-value">{html.escape(name)}, {html.escape(value(result))}</p>{shown}'


def frames(result: RunResult) -> str:
    """Each frame as pandas prints it, under the feature group that answered it and for whom."""
    steps = [step for step in result.plan if step.requested_feature_names]
    cells = []
    for step, frame in zip(steps, result, strict=True):
        customer = option(step, CUSTOMER)
        group = step.feature_group.__name__ if step.feature_group else "?"
        label = f"{group}, {f'customer {customer}' if customer is not None else 'every customer'}"
        cells.append(
            f'<div><p class="retail-label">{html.escape(label)}</p><pre><code>{html.escape(repr(frame))}</code></pre></div>'
        )
    return f'<div class="retail-frames">{"".join(cells)}</div>'
