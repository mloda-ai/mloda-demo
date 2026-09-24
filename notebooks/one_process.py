import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", css_file="physical_ai.css")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    from mloda_demo.one_process import mloda_run

    return mloda_run, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # One Pipeline, Any Device
    ## Swap the reader, not the pipeline

    Tom Kaltofen, mloda.ai
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## One process, two implementations""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div class="boxes">
      <div class="box"><h3>Data engineering</h3><p>batch pipeline</p><p>semantic layer</p></div>
      <div class="box"><h3>ML</h3><p>offline feature</p><p>online feature</p></div>
      <div class="box"><h3>AI agents</h3><p>index, embeddings</p><p>tool call</p></div>
      <div class="box"><h3>Robotics</h3><p>log replay, sim</p><p>on-board</p></div>
    </div>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Why: two pipelines""")
    return


@app.cell
def _(mloda_run):
    mloda_run(["tum_stop", "redwood_stop"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## One definition, small readers""")
    return


@app.cell
def _(mloda_run):
    mloda_run(["tum__metres__nearest__stop", "redwood__metres__nearest__stop"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## ML: offline, online""")
    return


@app.cell
def _(mloda_run):
    mloda_run(["history__monthly_payment", "applicant__monthly_payment"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Semantic layer: two departments, one revenue""")
    return


@app.cell
def _(mloda_run):
    mloda_run(["finance__revenue__per_customer", "sales__revenue__per_customer"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Robotics: log, sim, synthetic""")
    return


@app.cell
def _(mloda_run):
    mloda_run(["tum__metres__nearest__stop", "redwood__metres__nearest__stop", "sim__metres__nearest__stop"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""## Open""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <div class="close">
      <div>
        <p>Pipelines you can share.</p>
        <p>Runs you can reproduce.</p>
        <p>A process you certify once.</p>
        <p class="muted">Early days. A bet.</p>
      </div>
      <div class="qr">
        <svg viewBox="-4 -4 37 37" shape-rendering="crispEdges" role="img" aria-label="QR code: github.com/mloda-ai/mloda-demo">
          <rect x="-4" y="-4" width="37" height="37" fill="#FFFFFF"/>
          <path stroke="#0E120F" d="M0 0.5h7m1 0h4m2 0h1m2 0h2m3 0h7m-29 1h1m5 0h1m4 0h2m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m5 0h1m-29 1h1m1 0h3m1 0h1m1 0h1m2 0h3m1 0h1m1 0h2m3 0h1m1 0h3m1 0h1m-29 1h1m1 0h3m1 0h1m3 0h2m1 0h2m2 0h3m2 0h1m1 0h3m1 0h1m-29 1h1m1 0h3m1 0h1m2 0h1m4 0h1m2 0h2m3 0h1m1 0h3m1 0h1m-29 1h1m5 0h1m1 0h2m1 0h1m1 0h2m7 0h1m5 0h1m-29 1h7m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m1 0h7m-19 1h3m1 0h1m1 0h1m1 0h1m1 0h1m-21 1h1m1 0h1m3 0h2m1 0h2m2 0h4m1 0h1m1 0h1m2 0h1m2 0h1m1 0h1m-28 1h3m1 0h1m1 0h2m1 0h1m1 0h1m2 0h3m1 0h5m3 0h2m-27 1h2m2 0h1m1 0h1m4 0h3m1 0h1m1 0h1m2 0h5m1 0h1m-28 1h3m1 0h1m3 0h2m5 0h1m1 0h1m1 0h2m2 0h2m-23 1h1m2 0h1m1 0h2m1 0h4m1 0h1m3 0h1m1 0h2m4 0h1m-29 1h1m1 0h4m1 0h1m2 0h4m1 0h1m3 0h2m1 0h2m3 0h2m-29 1h3m3 0h1m1 0h5m2 0h1m1 0h3m1 0h4m3 0h1m-29 1h1m2 0h1m3 0h1m1 0h2m2 0h1m4 0h3m1 0h2m-24 1h2m2 0h1m1 0h1m1 0h2m3 0h1m1 0h2m3 0h3m5 0h1m-27 1h1m2 0h1m4 0h3m2 0h3m1 0h1m2 0h2m2 0h3m-29 1h3m1 0h6m1 0h1m1 0h7m4 0h2m2 0h1m-22 1h4m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m1 0h1m-25 1h3m3 0h1m2 0h1m1 0h1m1 0h4m1 0h1m1 0h6m1 0h1m-20 1h2m1 0h7m2 0h1m3 0h3m1 0h1m-29 1h7m1 0h1m2 0h1m3 0h1m4 0h1m1 0h1m1 0h1m3 0h1m-29 1h1m5 0h1m4 0h3m1 0h2m1 0h3m3 0h1m3 0h1m-29 1h1m1 0h3m1 0h1m3 0h2m1 0h1m2 0h1m2 0h7m2 0h1m-29 1h1m1 0h3m1 0h1m4 0h1m3 0h3m1 0h3m2 0h3m1 0h1m-29 1h1m1 0h3m1 0h1m1 0h2m2 0h6m3 0h1m2 0h1m2 0h2m-29 1h1m5 0h1m3 0h1m1 0h3m1 0h1m1 0h1m1 0h1m2 0h3m-26 1h7m1 0h3m5 0h1m3 0h1m3 0h2m2 0h1"/>
        </svg>
        <span>github.com/mloda-ai/mloda-demo</span>
      </div>
    </div>
    """)
    return


if __name__ == "__main__":
    app.run()
