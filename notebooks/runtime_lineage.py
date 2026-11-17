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
    from mloda.community.extenders.openlineage import OpenLineageExtender
    from mloda.community.extenders.otel.otel_extender import OtelExtender
    from mloda.community.feature_groups.experimental.german_ledger.skr import ChartConflict
    from mloda.user import Feature, mloda

    from mloda_demo import runtime_lineage as demo

    # The local details, named once: the clients' books, the charts the office can read, where the hooks send.
    client_b = str(demo.CLIENT_B)
    client_b_corrected = str(demo.CLIENT_B_CORRECTED)
    reads_skr04, reads_skr03 = demo.READS_SKR04, demo.READS_SKR03
    recorded = demo.telemetry()
    traces, lineage = recorded.tracer_provider, recorded.lineage
    metrics = recorded.metrics  # empty; the OtelMetricsExtender with the stack
    office = {"compute_frameworks": ["PyArrowTable"], "data_access_collection": demo.BOOKS}
    show = functools.partial(demo.show, telemetry=recorded)
    return (
        ChartConflict,
        Feature,
        OpenLineageExtender,
        OtelExtender,
        client_b,
        client_b_corrected,
        lineage,
        metrics,
        mloda,
        mo,
        office,
        reads_skr03,
        reads_skr04,
        show,
        traces,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The case: two numbers, no suspect

    | Question | Client says | Monitoring |
    |---|---|---|
    | Open receivables of client B? | 7,500.00 | all green |

    The number goes into the 2025 closing.
    """)
    return


@app.cell
def _(
    Feature,
    OpenLineageExtender,
    OtelExtender,
    client_b,
    lineage,
    metrics,
    mloda,
    office,
    reads_skr04,
    show,
    traces,
):
    hooks = {OtelExtender(tracer_provider=traces), OpenLineageExtender(client=lineage), *metrics}
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
    return (hooks,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The witness: what the hooks recorded
    """)
    return


@app.cell
def _(Feature, client_b, hooks, mloda, office, reads_skr04, show):
    # The same question, now with what the two hooks recorded.
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
    ## The suspect: the chart of accounts
    """)
    return


@app.cell
def _(Feature, client_b, hooks, mloda, office, reads_skr03, show):
    # Client B's same file, read with the other chart.
    show(
        mloda.run_all(
            [Feature("receivables__sources", options={"DatevExtfReader": client_b})],
            plugin_collector=reads_skr03,
            function_extender=hooks,
            **office,
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The fix: make it fail loudly
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


if __name__ == "__main__":
    app.run()
