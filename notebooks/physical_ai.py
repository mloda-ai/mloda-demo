import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="physical_ai.css")


@app.cell(hide_code=True)
def _():
    import html
    import inspect

    import marimo as mo

    from mloda_demo.physical_ai.clip import CLIP_DIR, frame_index, load_rgb
    from mloda_demo.physical_ai.plot import frame_view
    from mloda_demo.physical_ai.readers import DepthPng, TumDepth
    from mloda_demo.physical_ai.runner import at_frame, closest_frame, run
    from mloda_demo.physical_ai.trace import trace_html

    # Run headless (tests, `python notebooks/physical_ai.py`) with every gate open.
    SCRIPT = mo.app_meta().mode == "script"
    frames = frame_index(CLIP_DIR)
    return (
        CLIP_DIR,
        DepthPng,
        SCRIPT,
        TumDepth,
        at_frame,
        closest_frame,
        frame_view,
        frames,
        html,
        inspect,
        load_rgb,
        mo,
        run,
        trace_html,
    )


@app.cell(hide_code=True)
def _(CLIP_DIR, at_frame, frame_view, frames, load_rgb):
    rendered = {}

    def view(result, frame):
        key = (result.label, frame)
        if key not in rendered:
            row = at_frame(result, frame)
            rgb = load_rgb(CLIP_DIR / frames.loc[frames["frame"] == frame, "rgb"].iloc[0])
            nearest, stop, t_s = float(row["nearest_ahead_m"]), bool(row["stop"]), float(row["t_s"])
            rendered[key] = frame_view(rgb, row["depth_m"], nearest, stop, t_s)
        return rendered[key]

    return (view,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The robot and the chair
    Freiburg. A Pioneer robot with a Kinect on top. Stop below 1 m.
    """)
    return


@app.cell
def _(DepthPng, TumDepth):
    reader = DepthPng  # the TUM reader: TumDepth
    return (reader,)


@app.cell(hide_code=True)
def _(reader, run):
    result = run(reader)
    return (result,)


@app.cell(hide_code=True)
def _(frames, mo, reader):
    # Recreated on a reader swap, so the replay starts again at the first frame.
    replay = mo.ui.slider(
        steps=frames["frame"].tolist(), value=0, label=f"frame ({reader.label})", show_value=True, full_width=True
    )
    mo.hstack([replay])
    return (replay,)


@app.cell(hide_code=True)
def _(replay, result, view):
    view(result, replay.value)
    return


@app.cell(hide_code=True)
def _(mo, replay, result, trace_html):
    mo.vstack([mo.md("## The receipt"), mo.Html(trace_html(result, replay.value))])
    return


@app.cell(hide_code=True)
def _(TumDepth, html, inspect, mo):
    source = html.escape(inspect.getsource(TumDepth))
    source = source.replace("scale = 5000  # per metre", "<mark>scale = 5000  # per metre</mark>", 1)
    mo.vstack([mo.md("## Swap the reader"), mo.Html(f'<pre class="trace">{source}</pre>')])
    return


@app.cell(hide_code=True)
def _(mo):
    rerun_broken = mo.ui.run_button(label="rerun the broken setup")
    mo.vstack([mo.md("## Make it a check: the conversion needs a declared scale"), rerun_broken])
    return (rerun_broken,)


@app.cell(hide_code=True)
def _(SCRIPT, DepthPng, mo, rerun_broken, run):
    mo.stop(not (rerun_broken.value or SCRIPT))
    checked = run(DepthPng, check=True)
    mo.callout(mo.md(checked.error or "no error"), kind="danger" if checked.error else "success")
    return (checked,)


@app.cell(hide_code=True)
def _(closest_frame, result):
    # Headless runs report the chair frame; on stage the slider does.
    chair = closest_frame(result)
    return (chair,)


if __name__ == "__main__":
    app.run()
