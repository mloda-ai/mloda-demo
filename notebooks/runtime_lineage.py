# /// script
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# [tool.marimo.display]
# reference_highlighting = false
# code_lens = false
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(
    width="medium",
    layout_file="layouts/runtime_lineage.slides.json",
    css_file="retail.css",
    html_head_file="retail.head.html",
)


@app.cell(hide_code=True)
def _():
    import functools

    import marimo as mo
    from mloda.community.feature_groups.experimental.german_ledger.skr import ChartConflict
    from mloda.user import Feature, mloda

    from mloda_demo import runtime_lineage as demo

    # The local details, named once: the clients' books, the charts the office can read, the hooks.
    client_a, client_b, client_c = str(demo.CLIENT_A), str(demo.CLIENT_B), str(demo.CLIENT_C)
    client_b_corrected = str(demo.CLIENT_B_CORRECTED)
    reads_skr04, reads_skr03 = demo.READS_SKR04, demo.READS_SKR03
    recorded = demo.telemetry()
    hooks = set(recorded.extenders)  # OtelExtender and OpenLineageExtender; OtelMetricsExtender with the stack
    office = {"compute_frameworks": ["PyArrowTable"], "data_access_collection": demo.BOOKS}
    show = functools.partial(demo.show, telemetry=recorded)
    return (
        ChartConflict,
        Feature,
        client_a,
        client_b,
        client_b_corrected,
        client_c,
        hooks,
        mloda,
        mo,
        office,
        reads_skr03,
        reads_skr04,
        show,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The case

    | The run | The number |
    |---|---|
    | Green | Wrong |
    """)
    return


@app.cell
def _(Feature, client_b, hooks, mloda, office, reads_skr04, show):
    # What are client B's open receivables?
    show(
        mloda.run_all(
            [Feature("receivables__sources", options={"DatevExtfReader": client_b})],
            plugin_collector=reads_skr04,
            function_extender=hooks,
            **office,
        ),
        evidence=False,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 One answer, one ID

    | You know | The small change |
    |---|---|
    | A trace | Every span is a data step with a lineage twin |
    """)
    return


@app.cell
def _(Feature, client_a, hooks, mloda, office, reads_skr04, show):
    # What was client A's revenue in 2025?
    show(
        mloda.run_all(
            [Feature("revenue__sources", options={"DatevExtfReader": client_a})],
            plugin_collector=reads_skr04,
            function_extender=hooks,
            **office,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 Another request, another path

    | You know | The small change |
    |---|---|
    | Lineage as a fixed map | It belongs to the request |
    """)
    return


@app.cell
def _(Feature, client_c, hooks, mloda, office, reads_skr04, show):
    # The same question for client C, whose books arrive as a tax audit export.
    show(
        mloda.run_all(
            [Feature("revenue__sources", options={"GdpduReader": client_c})],
            plugin_collector=reads_skr04,
            function_extender=hooks,
            **office,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 The wrong number

    | You know | The small change |
    |---|---|
    | Red spans | A green span carries evidence |
    """)
    return


@app.cell
def _(Feature, client_b, hooks, mloda, office, reads_skr04, show):
    # Client B again: where did this number come from?
    show(
        mloda.run_all(
            [Feature("receivables__sources", options={"DatevExtfReader": client_b})],
            plugin_collector=reads_skr04,
            function_extender=hooks,
            **office,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 Make it fail loudly

    | You know | The small change |
    |---|---|
    | Alerts on failures | Wrong data became a failure |
    """)
    return


@app.cell
def _(ChartConflict, Feature, client_b_corrected, hooks, mloda, office, reads_skr04, show):
    # Client B sends a corrected export that states its chart.
    try:
        _answer = mloda.run_all(
            [Feature("receivables__sources", options={"DatevExtfReader": client_b_corrected})],
            plugin_collector=reads_skr04,
            function_extender=hooks,
            **office,
        )
    except ChartConflict as _refusal:
        _answer = _refusal
    show(_answer)
    return


@app.cell
def _(Feature, client_b_corrected, hooks, mloda, office, reads_skr03, show):
    # The office sets client B up with its own chart.
    show(
        mloda.run_all(
            [Feature("receivables__sources", options={"DatevExtfReader": client_b_corrected})],
            plugin_collector=reads_skr03,
            function_extender=hooks,
            **office,
        )
    )
    return


if __name__ == "__main__":
    app.run()
