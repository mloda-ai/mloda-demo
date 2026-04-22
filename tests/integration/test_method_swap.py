"""Integration: EpsilonPlus vs Gradient must both run and produce different attributions."""

from __future__ import annotations

import numpy as np
import pytest

from mloda_demo.feature_groups.classifier.artifact import MODEL_STATE_PATH, load_artifact
from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import _ensure_artifact
from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS
from mloda_demo.feature_groups.inputs.applications_fg import ApplicationsFG
from mloda_demo.feature_groups.inputs.financials_fg import FinancialsFG
from mloda_demo.feature_groups.inputs.questionnaire_fg import QuestionnaireFG
from mloda_demo.xai.attribution.gradient_attribution import GradientAttributionFeatureGroup
from mloda_demo.xai.attribution.zennit_attribution import ZennitAttributionFeatureGroup


@pytest.mark.slow
def test_method_swap_produces_different_attributions() -> None:
    _ensure_artifact()
    apps = ApplicationsFG.calculate_feature(None, None)
    fin = FinancialsFG.calculate_feature(None, None)
    qa = QuestionnaireFG.calculate_feature(None, None)
    merged = apps.merge(fin, on="customer_id").merge(qa, on="customer_id")

    artifact = load_artifact()
    X = artifact.encoder.encode(merged[FEATURE_COLUMNS])
    model = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))

    rel_eps = ZennitAttributionFeatureGroup._compute_attributions(model, X, "EpsilonPlus", None)
    rel_grad = GradientAttributionFeatureGroup._compute_attributions(model, X, "Gradient", None)

    assert rel_eps.shape == rel_grad.shape == (5, len(FEATURE_COLUMNS))
    assert float(np.abs(rel_eps).sum()) > 0.0
    assert float(np.abs(rel_grad).sum()) > 0.0
    assert not np.allclose(rel_eps, rel_grad), "EpsilonPlus and Gradient should produce different attributions"
