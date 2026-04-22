"""Unit tests mirroring the 5 acts of the applydata 2026 live demo.

Each act in the handbook (`demo/applydata_handbook.md`) corresponds to one
test here. These exercise the deterministic substrate the agent-on-top
relies on: same input -> same output, every time. No subprocess, no LLM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mloda.user import Feature
from mloda.user import mloda as mloda_api

from mloda_demo.cli import (
    DEFAULT_XAI_METHOD,
    _get_demo_fgs,
    _get_input_links,
)
from mloda_demo.feature_groups.classifier.artifact import MODEL_STATE_PATH, load_artifact
from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import _ensure_artifact
from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS
from mloda_demo.feature_groups.inputs.applications_fg import ApplicationsFG
from mloda_demo.feature_groups.inputs.financials_fg import FinancialsFG
from mloda_demo.feature_groups.inputs.questionnaire_fg import QuestionnaireFG
from mloda_demo.xai.attribution.gradient_attribution import GradientAttributionFeatureGroup
from mloda_demo.xai.attribution.zennit_attribution import ZennitAttributionFeatureGroup


KNOWN_CUSTOMER = "app-customer-c"


# Act 1 -- Discover -----------------------------------------------------------


def test_act1_discover_lists_demo_fgs() -> None:
    """The demo FGs are importable and registered for discovery (active backend only)."""
    names = {fg.__name__ for fg in _get_demo_fgs()}
    expected = {
        "ApplicationsFG",  # JSON backend (default)
        "FinancialsFG",
        "QuestionnaireFG",
        "CreditRiskClassifierFG",
        "ZennitAttributionFeatureGroup",
        "GradientAttributionFeatureGroup",
    }
    assert expected.issubset(names)


def test_act1_input_fgs_expose_their_columns_via_input_data() -> None:
    """Discover surface for the input FGs comes from DataCreator(COLUMNS)."""
    apps_cols = set(ApplicationsFG.input_data().feature_names)  # type: ignore[union-attr]
    fin_cols = set(FinancialsFG.input_data().feature_names)  # type: ignore[union-attr]
    qa_cols = set(QuestionnaireFG.input_data().feature_names)  # type: ignore[union-attr]
    assert {"customer_id", "credit_amount", "purpose"} <= apps_cols
    assert {"duration", "age"} <= fin_cols
    assert {"checking_status", "employment"} <= qa_cols


# Act 2 -- Pull row through mloda.run_all -------------------------------------


@pytest.mark.slow
def test_act2_run_all_pulls_input_features_from_three_fgs() -> None:
    """mloda.run_all resolves features across the 3 root FGs and returns them."""
    features = [Feature.not_typed(name) for name in ("duration", "credit_amount", "checking_status")]
    results = mloda_api.run_all(features=features, compute_frameworks=["PandasDataFrame"])
    assert len(results) == 3
    cols = {col for r in results for col in r.columns}
    assert {"duration", "credit_amount", "checking_status"} <= cols
    for r in results:
        assert len(r) == 5


# Act 3 -- Predict via mloda.run_all (Links chain the input FGs) --------------


@pytest.mark.slow
def test_act3_predict_for_known_customer() -> None:
    """credit_risk resolves through CreditRiskClassifierFG, with Links wiring inputs."""
    results = mloda_api.run_all(
        features=[Feature.not_typed("credit_risk")],
        compute_frameworks=["PandasDataFrame"],
        links=_get_input_links(),
    )
    assert len(results) == 1
    df = results[0]
    assert "credit_risk" in df.columns
    assert len(df) == 5
    for v in df["credit_risk"]:
        assert v in (0, 1)


# Act 3b -- Explain (attribution) ---------------------------------------------


@pytest.fixture(scope="module")
def known_customer_encoded() -> np.ndarray:  # type: ignore[type-arg]
    """Encoded feature row for the known customer, ready for attribution."""
    _ensure_artifact()
    apps = ApplicationsFG.calculate_feature(None, None)
    fin = FinancialsFG.calculate_feature(None, None)
    qa = QuestionnaireFG.calculate_feature(None, None)
    merged: pd.DataFrame = apps.merge(fin, on="customer_id").merge(qa, on="customer_id")
    row = merged[merged["customer_id"] == KNOWN_CUSTOMER]
    artifact = load_artifact()
    assert artifact is not None
    return artifact.encoder.encode(row[FEATURE_COLUMNS])


@pytest.mark.slow
def test_act3b_attribution_for_known_customer_epsilonplus(
    known_customer_encoded: np.ndarray,  # type: ignore[type-arg]
) -> None:
    model = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    relevance = ZennitAttributionFeatureGroup._compute_attributions(
        model, known_customer_encoded, DEFAULT_XAI_METHOD, None
    )
    assert relevance.shape == (1, len(FEATURE_COLUMNS))
    assert np.isfinite(relevance).all()
    assert float(np.abs(relevance).sum()) > 0.0


# Act 4 -- Swap method --------------------------------------------------------


@pytest.mark.slow
def test_act4_method_swap_changes_relevance_but_both_run(
    known_customer_encoded: np.ndarray,  # type: ignore[type-arg]
) -> None:
    z_model = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    g_model = GradientAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    rel_eps = ZennitAttributionFeatureGroup._compute_attributions(z_model, known_customer_encoded, "EpsilonPlus", None)
    rel_grad = GradientAttributionFeatureGroup._compute_attributions(g_model, known_customer_encoded, "Gradient", None)
    assert rel_eps.shape == rel_grad.shape
    assert not np.allclose(rel_eps, rel_grad)


# Act 5 -- Determinism --------------------------------------------------------


@pytest.mark.slow
def test_act5_predict_is_bit_identical_across_runs() -> None:
    """The thesis: same prompt twice -> same numbers twice."""
    first = mloda_api.run_all(
        features=[Feature.not_typed("credit_risk")],
        compute_frameworks=["PandasDataFrame"],
        links=_get_input_links(),
    )
    second = mloda_api.run_all(
        features=[Feature.not_typed("credit_risk")],
        compute_frameworks=["PandasDataFrame"],
        links=_get_input_links(),
    )
    assert list(first[0]["credit_risk"]) == list(second[0]["credit_risk"])


@pytest.mark.slow
def test_act5_attribution_is_bit_identical_across_runs(
    known_customer_encoded: np.ndarray,  # type: ignore[type-arg]
) -> None:
    model = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    rel_a = ZennitAttributionFeatureGroup._compute_attributions(model, known_customer_encoded, DEFAULT_XAI_METHOD, None)
    rel_b = ZennitAttributionFeatureGroup._compute_attributions(model, known_customer_encoded, DEFAULT_XAI_METHOD, None)
    assert np.array_equal(rel_a, rel_b)
