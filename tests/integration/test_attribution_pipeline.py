"""Integration: end-to-end Zennit attribution from merged customers to relevance matrix."""

from __future__ import annotations

import numpy as np
import pytest

from mloda_demo.feature_groups.classifier.artifact import MODEL_STATE_PATH, load_artifact
from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import _ensure_artifact
from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS
from mloda_demo.feature_groups.inputs.applications_fg import ApplicationsFG
from mloda_demo.feature_groups.inputs.financials_fg import FinancialsFG
from mloda_demo.feature_groups.inputs.questionnaire_fg import QuestionnaireFG
from mloda_demo.xai.attribution.zennit_attribution import ZennitAttributionFeatureGroup


@pytest.fixture(scope="module")
def merged_rows():  # type: ignore[no-untyped-def]
    _ensure_artifact()
    apps = ApplicationsFG.calculate_feature(None, None)
    fin = FinancialsFG.calculate_feature(None, None)
    qa = QuestionnaireFG.calculate_feature(None, None)
    return apps.merge(fin, on="customer_id").merge(qa, on="customer_id")


@pytest.mark.slow
def test_epsilon_plus_attribution_shape(merged_rows) -> None:  # type: ignore[no-untyped-def]
    artifact = load_artifact()
    assert artifact is not None
    X = artifact.encoder.encode(merged_rows[FEATURE_COLUMNS])
    model = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    relevance = ZennitAttributionFeatureGroup._compute_attributions(model, X, "EpsilonPlus", None)

    assert relevance.shape == (5, len(FEATURE_COLUMNS))
    assert np.isfinite(relevance).all()
    assert float(np.abs(relevance).sum()) > 0.0


@pytest.mark.slow
def test_attribution_target_class_changes_relevance(merged_rows) -> None:  # type: ignore[no-untyped-def]
    artifact = load_artifact()
    X = artifact.encoder.encode(merged_rows[FEATURE_COLUMNS])
    model = ZennitAttributionFeatureGroup._load_model(str(MODEL_STATE_PATH))
    rel_class_0 = ZennitAttributionFeatureGroup._compute_attributions(model, X, "EpsilonPlus", 0)
    rel_class_1 = ZennitAttributionFeatureGroup._compute_attributions(model, X, "EpsilonPlus", 1)
    assert not np.allclose(rel_class_0, rel_class_1)
