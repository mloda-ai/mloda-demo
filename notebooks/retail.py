# /// script
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# [tool.marimo.display]
# reference_highlighting = false
# code_lens = false
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/retail.slides.json",
    css_file="retail.css",
    html_head_file="retail.head.html",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from mloda.user import Feature, FeatureResolutionError, Options, get_feature_group_docs, mloda

    from mloda_demo.retail import LEDGER, ensure_store, show

    # The local details, named once: which source, which point in time, which customer.
    ledger, checkout_time = str(LEDGER), "2010-08-31 15:37"
    every_customer = Options(group={"ShopLedger": ledger, "as_of": checkout_time})
    one_customer = Options(group={"ShopLedger": ledger, "as_of": checkout_time, "customer": 14045})
    at_checkout = Options(group={"as_of": checkout_time, "customer": 14045})
    from_marketing = Options(group={"MarketingExport": ledger, "as_of": checkout_time, "customer": 14045})
    ensure_store()  # the stand-in store holds what last night's batch computed
    return (
        Feature,
        FeatureResolutionError,
        at_checkout,
        every_customer,
        from_marketing,
        get_feature_group_docs,
        mloda,
        mo,
        one_customer,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 Find what exists

    | You know | The small change |
    |---|---|
    | The registry | It is a package |
    """)
    return


@app.cell
def _(get_feature_group_docs, show):
    # Search the installed packages: which numbers are called spend?
    show(get_feature_group_docs(search="spend"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 Understand a number

    | You know | The small change |
    |---|---|
    | The feature definition | The definition itself runs |
    """)
    return


@app.cell
def _(Feature, mloda, one_customer, show):
    # What does net_spend_30d mean?
    show(mloda.run_all([Feature("net_spend_30d", one_customer, feature_group="NetSpend30d")]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 Trust it

    | You know | The small change |
    |---|---|
    | Schema and validation | A contract |
    """)
    return


@app.cell
def _(Feature, FeatureResolutionError, from_marketing, mloda, show):
    # Can I compute net_spend_30d from marketing's export?
    try:
        answer = mloda.run_all([Feature("net_spend_30d", from_marketing, feature_group="NetSpend30d")])
    except FeatureResolutionError as refusal:
        answer = refusal
    show(answer)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Use it

    | You know | The small change |
    |---|---|
    | Offline and online | Declarative |
    """)
    return


@app.cell
def _(Feature, at_checkout, every_customer, mloda, one_customer, show):
    # Which number did the checkout use? Each consumer declares what it wants.
    show(
        mloda.run_all(
            [
                Feature("net_spend_30d", every_customer, feature_group="NetSpend30d"),  # training
                Feature("net_spend_30d", at_checkout, feature_group="FeatureStore"),  # checkout
                Feature("net_spend_30d", one_customer, feature_group="NetSpend30d"),  # agent
            ]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 Ask something new

    | You know | The small change |
    |---|---|
    | On-demand features | Nobody registered this combination |
    """)
    return


@app.cell
def _(Feature, mloda, one_customer, show):
    # What did this customer spend in the 7 days before the last return?
    show(mloda.run_all([Feature("line_value__7d_before__last_return", one_customer, feature_group="DaysBefore")]))
    return


if __name__ == "__main__":
    app.run()
