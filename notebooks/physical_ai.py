import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="physical_ai.css")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from mloda_demo.physical_ai.definition import ASSUMED_UNITS, FAULTY_UNITS
    from mloda_demo.physical_ai.plot import top_down
    from mloda_demo.physical_ai.readers import DepthReaderA, DepthReaderB
    from mloda_demo.physical_ai.runner import (
        FEATURES,
        FOCUS_OBJECT,
        all_passed,
        compare,
        n_table,
        nearest_points,
        run,
    )
    from mloda_demo.physical_ai.scene import TIMESTAMPS
    from mloda_demo.physical_ai.trace import trace_html

    # Run headless (tests, `python notebooks/physical_ai.py`) with every gate open.
    SCRIPT = mo.app_meta().mode == "script"
    return (
        ASSUMED_UNITS,
        DepthReaderA,
        DepthReaderB,
        FAULTY_UNITS,
        FEATURES,
        FOCUS_OBJECT,
        SCRIPT,
        TIMESTAMPS,
        all_passed,
        compare,
        mo,
        n_table,
        nearest_points,
        run,
        top_down,
        trace_html,
    )


@app.cell(hide_code=True)
def _(FOCUS_OBJECT, all_passed, mo, n_table, nearest_points, top_down):
    def badge(runs):
        columns = ", ".join(runs)
        if all_passed(runs.values()):
            return mo.callout(mo.md(f"**all runs passed**: {columns}"), kind="success")
        return mo.callout(mo.md(f"**a run failed**: {columns}"), kind="danger")

    def table_view(runs):
        html = n_table(runs).to_html(border=0)
        html = html.replace("<td>NO</td>", '<td class="brake">NO</td>')
        return mo.Html(f'<div style="font-size:1.6em">{html}</div>')

    def comparison(runs):
        offline, device = runs["offline"], list(runs.values())[-1]
        ghost = nearest_points(offline).loc[FOCUS_OBJECT]
        points = nearest_points(device)
        moved = abs(points.loc[FOCUS_OBJECT, "nearest_distance_m"] - ghost["nearest_distance_m"]) > 1e-3
        figure = top_down(points, ghost=(ghost["x_m"], ghost["y_m"]) if moved else None, title=device.label)
        return mo.hstack([figure, table_view(runs)], align="center")

    return badge, comparison, table_view


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Offline: Device A logs become training data
    Synthetic devices, no hardware. Brake below 2 m.
    """)
    return


@app.cell
def _(DepthReaderA, nearest_points, run, top_down):
    offline = run(DepthReaderA)
    top_down(nearest_points(offline), title="offline: device A logs")
    return


@app.cell(hide_code=True)
def _(TIMESTAMPS, mo):
    frame = mo.ui.slider(0, len(TIMESTAMPS) - 1, value=len(TIMESTAMPS) - 1, label="replay frame", show_value=True)
    mo.vstack([mo.md("## Online: the same definition, frame by frame (a Python replay)"), frame])
    return (frame,)


@app.cell
def _(DepthReaderA, frame, nearest_points, run, top_down):
    online = run(DepthReaderA, label="online", replay_until=frame.value)
    top_down(nearest_points(online, frame.value), title=f"online replay, frame {frame.value}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## n: the new vendor
    """)
    return


@app.cell
def _(DepthReaderA, DepthReaderB):
    reader = DepthReaderA  # the new vendor: DepthReaderB
    return (reader,)


@app.cell(hide_code=True)
def _(compare, fault, reader):
    runs = compare(reader, fault=fault)
    return (runs,)


@app.cell(hide_code=True)
def _(mo):
    show_badge = mo.ui.run_button(label="show result")
    show_plot = mo.ui.run_button(label="compare")
    mo.hstack([show_badge, show_plot], justify="start")
    return show_badge, show_plot


@app.cell(hide_code=True)
def _(SCRIPT, badge, mo, runs, show_badge):
    mo.stop(not (show_badge.value or SCRIPT))
    badge(runs)
    return


@app.cell(hide_code=True)
def _(SCRIPT, comparison, mo, runs, show_plot):
    mo.stop(not (show_plot.value or SCRIPT))
    comparison(runs)
    return


@app.cell(hide_code=True)
def _(mo, runs, trace_html):
    device_run = list(runs.values())[-1]
    mo.vstack([mo.md("## Trace: where the number came from"), mo.Html(trace_html(device_run))])
    return


@app.cell(hide_code=True)
def _(ASSUMED_UNITS, FAULTY_UNITS, mo):
    fix = mo.ui.switch(label="apply the fix")
    diff = f"```diff\n # DepthToMetres: unit assumed per raw encoding\n-{FAULTY_UNITS}\n+{ASSUMED_UNITS}\n```"
    mo.vstack([mo.md(diff), fix])
    return (fix,)


@app.cell
def _(fix):
    fault = not fix.value
    return (fault,)


@app.cell(hide_code=True)
def _(comparison, runs):
    comparison(runs)
    return


@app.cell(hide_code=True)
def _(mo):
    rerun_broken = mo.ui.run_button(label="rerun the broken setup")
    mo.vstack([mo.md("## Make it a check: the conversion declares the unit it assumes"), rerun_broken])
    return (rerun_broken,)


@app.cell
def _(DepthReaderB, run):
    checked = run(DepthReaderB, fault=True, check_units=True)
    return (checked,)


@app.cell(hide_code=True)
def _(SCRIPT, checked, mo, rerun_broken):
    mo.stop(not (rerun_broken.value or SCRIPT))
    mo.callout(mo.md(checked.error or "no error"), kind="danger" if checked.error else "success")
    return


@app.cell(hide_code=True)
def _(mo):
    ask = mo.ui.run_button(label="ask")
    mo.vstack([mo.md("## Next question: which object is approaching?"), ask])
    return (ask,)


@app.cell
def _(FEATURES, compare, fault, reader):
    asked = compare(reader, fault=fault, features=(*FEATURES, "approaching"))
    return (asked,)


@app.cell(hide_code=True)
def _(SCRIPT, ask, asked, mo, table_view):
    mo.stop(not (ask.value or SCRIPT))
    table_view(asked)
    return


if __name__ == "__main__":
    app.run()
