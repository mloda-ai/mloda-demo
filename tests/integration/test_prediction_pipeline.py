"""Integration: full pipeline from three input FGs to per-customer prediction."""

from __future__ import annotations

import pytest

from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import CreditRiskClassifierFG
from mloda_demo.feature_groups.inputs.applications_fg import ApplicationsFG
from mloda_demo.feature_groups.inputs.financials_fg import FinancialsFG
from mloda_demo.feature_groups.inputs.questionnaire_fg import QuestionnaireFG


@pytest.mark.slow
def test_full_pipeline_predicts_five_customers() -> None:
    apps = ApplicationsFG.calculate_feature(None, None)
    fin = FinancialsFG.calculate_feature(None, None)
    qa = QuestionnaireFG.calculate_feature(None, None)

    merged = apps.merge(fin, on="customer_id").merge(qa, on="customer_id")
    assert len(merged) == 5
    assert merged.shape[1] == 21

    result = CreditRiskClassifierFG.calculate_feature(merged, None)
    assert "credit_risk" in result
    predictions = result["credit_risk"]
    assert len(predictions) == 5
    for p in predictions:
        assert p in (0, 1)


@pytest.mark.slow
def test_predictions_are_deterministic_across_runs() -> None:
    apps = ApplicationsFG.calculate_feature(None, None)
    fin = FinancialsFG.calculate_feature(None, None)
    qa = QuestionnaireFG.calculate_feature(None, None)
    merged = apps.merge(fin, on="customer_id").merge(qa, on="customer_id")

    first = CreditRiskClassifierFG.calculate_feature(merged, None)
    second = CreditRiskClassifierFG.calculate_feature(merged, None)
    assert first["credit_risk"] == second["credit_risk"]
