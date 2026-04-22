"""applydata 2026 talk: mloda + MLP + Fraunhofer Zennit XAI.

Mixed structured + unstructured sources -> one row per customer -> explainable
credit-risk classifier -> feature attribution -> method cross-check.

Run with: marimo edit demo/applydata_credit_xai.py
"""

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.output.replace(
        mo.md(
            """
    # Deterministic Context for Non-Deterministic Agents

    ### mloda + Fraunhofer XAI, live on a mixed-source credit-risk pipeline

    | Act | What we add | What the audience sees |
    |---|---|---|
    | **1** | Three input feature groups + one classifier | Per-customer predictions from mixed JSON / Excel / Markdown |
    | **2** | `__attribution` suffix | Zennit EpsilonPlus heatmap showing which features drove each decision |
    | **3** | Swap one string: `EpsilonPlus` -> `Gradient` | Side-by-side heatmaps, same pipeline, different XAI method |

    The point: **the LLM / agent on top is non-deterministic. The context layer below it is deterministic.** Same mixed-source inputs always produce the same predictions and the same explanations.
    """
        )
    )
    return


@app.cell
def _():
    import json

    import pandas as pd

    return json, pd


@app.cell
def _():
    from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR

    return (DEMO_DATA_DIR,)


@app.cell
def _(mo):
    mo.md("""
    ## Three raw inputs, three shapes

    A real credit-risk pipeline pulls data from wherever the bank keeps it. We fake
    that here with three files per customer:

    - `applications.json` — structured workflow metadata (one JSON record per customer)
    - `financial_overview_<id>.xlsx` — numeric financial KPIs
    - `qa_<id>.md` — free-form questionnaire answers

    Every mloda FeatureGroup turns one of these into aligned, typed columns.
    """)
    return


@app.cell
def _(DEMO_DATA_DIR, json, pd):
    apps_json = json.loads((DEMO_DATA_DIR / "applications.json").read_text())
    apps_df = pd.DataFrame(apps_json)[["id", "applicant_name", "requested_amount", "purpose"]]
    return (apps_df,)


@app.cell
def _(apps_df, mo):
    mo.hstack([mo.md("**applications.json**"), mo.ui.table(apps_df, pagination=False, selection=None)])
    return


@app.cell
def _(DEMO_DATA_DIR, mo, pd):
    sample_xlsx = DEMO_DATA_DIR / "financial_overview_app-customer-a.xlsx"
    sample_md = DEMO_DATA_DIR / "qa_app-customer-a.md"
    sample_fin = pd.read_excel(sample_xlsx, sheet_name="financials", engine="openpyxl")
    sample_qa = sample_md.read_text()
    mo.hstack(
        [
            mo.vstack(
                [
                    mo.md("**financial_overview_app-customer-a.xlsx**"),
                    mo.ui.table(sample_fin, pagination=False, selection=None),
                ]
            ),
            mo.vstack([mo.md("**qa_app-customer-a.md**"), mo.md(f"```\n{sample_qa}\n```")]),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Act 1 — Predictions

    Three FeatureGroups extract the columns. The classifier FG trains an MLP on UCI
    German Credit (1 000 rows) once, caches the artifact, and predicts a class per
    customer. Nothing is glued together by hand: every feature comes from a
    registered FeatureGroup.
    """)
    return


@app.cell
def _():
    from mloda_demo.feature_groups.inputs.applications_fg import ApplicationsFG
    from mloda_demo.feature_groups.inputs.financials_fg import FinancialsFG
    from mloda_demo.feature_groups.inputs.questionnaire_fg import QuestionnaireFG

    apps = ApplicationsFG.calculate_feature(None, None)
    fin = FinancialsFG.calculate_feature(None, None)
    qa = QuestionnaireFG.calculate_feature(None, None)
    merged = apps.merge(fin, on="customer_id").merge(qa, on="customer_id")
    return (merged,)


@app.cell
def _(mo):
    mo.md("""
    **Merged customer rows (20 German Credit columns + customer_id):**
    """)
    return


@app.cell
def _(merged, mo):
    mo.ui.table(merged, pagination=False, selection=None)
    return


@app.cell
def _(merged):
    from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import CreditRiskClassifierFG

    prediction_result = CreditRiskClassifierFG.calculate_feature(merged, None)
    predictions = merged[["customer_id"]].copy()
    predictions["credit_risk"] = prediction_result["credit_risk"]
    predictions["decision"] = predictions["credit_risk"].map({1: "good", 0: "bad"})
    return (predictions,)


@app.cell
def _(mo, predictions):
    mo.ui.table(predictions, pagination=False, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Act 2 — Add `__attribution`

    One suffix change. The model is already saved to disk by the classifier FG.
    Zennit's EpsilonPlus composite runs LRP through every linear layer and returns
    a relevance score per feature.
    """)
    return


@app.cell
def _(merged, pd):
    from mloda_demo.feature_groups.classifier.artifact import MODEL_STATE_PATH, load_artifact
    from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS
    from mloda_demo.xai.attribution.zennit_attribution import ZennitAttributionFeatureGroup

    artifact = load_artifact()
    encoder = artifact.encoder
    X_encoded = encoder.encode(merged[FEATURE_COLUMNS])

    model_zennit = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    relevance_eps = ZennitAttributionFeatureGroup._compute_attributions(model_zennit, X_encoded, "EpsilonPlus", None)
    attribution_df = pd.DataFrame(relevance_eps, columns=FEATURE_COLUMNS)
    attribution_df.insert(0, "customer_id", merged["customer_id"].values)
    return FEATURE_COLUMNS, X_encoded, model_zennit, relevance_eps


@app.cell
def _(FEATURE_COLUMNS, plt, relevance_eps):
    fig_eps, ax_eps = plt.subplots(figsize=(12, 3.5))
    abs_max_eps = max(abs(relevance_eps.min()), abs(relevance_eps.max()))
    im_eps = ax_eps.imshow(relevance_eps, aspect="auto", cmap="RdBu_r", vmin=-abs_max_eps, vmax=abs_max_eps)
    ax_eps.set_xticks(range(len(FEATURE_COLUMNS)))
    ax_eps.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
    ax_eps.set_yticks(range(relevance_eps.shape[0]))
    ax_eps.set_yticklabels([f"customer {c}" for c in "abcde"])
    ax_eps.set_title("Zennit EpsilonPlus (LRP) attribution per customer")
    fig_eps.colorbar(im_eps, ax=ax_eps, label="Relevance")
    plt.tight_layout()
    fig_eps
    return


@app.cell
def _():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(mo):
    mo.md("""
    ## Act 3 — Swap one string

    Change `EpsilonPlus` to `Gradient`. Same pipeline, same model, same input
    columns. Different XAI method. A regulator asking for a second opinion can be
    answered with a one-line edit.
    """)
    return


@app.cell
def _(X_encoded, model_zennit):
    from mloda_demo.xai.attribution.gradient_attribution import GradientAttributionFeatureGroup

    relevance_grad = GradientAttributionFeatureGroup._compute_attributions(model_zennit, X_encoded, "Gradient", None)
    return (relevance_grad,)


@app.cell
def _(FEATURE_COLUMNS, plt, relevance_eps, relevance_grad):
    fig3, axes = plt.subplots(1, 2, figsize=(16, 3.5))
    abs_max = max(abs(relevance_eps).max(), abs(relevance_grad).max())
    for ax3, mat, title in zip(axes, [relevance_eps, relevance_grad], ["EpsilonPlus (LRP)", "Gradient"]):
        im3 = ax3.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
        ax3.set_xticks(range(len(FEATURE_COLUMNS)))
        ax3.set_xticklabels(FEATURE_COLUMNS, rotation=45, ha="right")
        ax3.set_yticks(range(mat.shape[0]))
        ax3.set_yticklabels([f"customer {c}" for c in "abcde"])
        ax3.set_title(title)
        fig3.colorbar(im3, ax=ax3, label="Relevance")
    plt.tight_layout()
    fig3
    return


@app.cell
def _(mo):
    mo.md("""
    ## The through-line

    1. Mixed-source inputs were read by three FeatureGroups with identical shape.
    2. A classifier FG trained once and is served from an artifact on every rerun.
    3. Adding a suffix turns a prediction into an explanation.
    4. Swapping a string turns one explanation method into another.

    **Same inputs, same predictions, same heatmap. Every time.** The LLM on top can
    take any conversational path; the pipeline below always produces the same
    audit-grade answer.
    """)
    return


if __name__ == "__main__":
    app.run()
