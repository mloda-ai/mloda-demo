import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    layout_file="layouts/retail.slides.json",
    css_file="physical_ai.css",
)


@app.cell(hide_code=True)
def _():
    from mloda_demo.retail import CUSTOMER, Agent, checkout, org

    agent = Agent()
    return CUSTOMER, agent, checkout, org


@app.cell(hide_code=True)
def _(org):
    org()
    return


@app.cell
def _(CUSTOMER, checkout):
    checkout(CUSTOMER)
    return


@app.cell
def _(agent):
    agent.ask("Why was I declined?")
    return


@app.cell
def _(agent):
    agent.ask("Which numbers are called spend?")
    return


@app.cell
def _(agent):
    agent.ask("What does net_spend_30d mean?")
    return


@app.cell
def _(agent):
    agent.ask("Can I compute net_spend_30d from the marketing export?")
    return


@app.cell
def _(agent):
    agent.ask("Which number did the checkout use?")
    return


@app.cell
def _(agent):
    agent.ask("Was the returned order the big one?")
    return


if __name__ == "__main__":
    app.run()
