from __future__ import annotations

import pandas as pd
import pytest

from mloda_demo.feature_groups.classifier.credit_risk_classifier_fg import CreditRiskClassifierFG
from mloda_demo.feature_groups.classifier.encoder import FEATURE_COLUMNS


def _fake_customer_row() -> dict[str, object]:
    return {
        "customer_id": "app-customer-a",
        "checking_status": "<0",
        "duration": 18,
        "credit_history": "existing paid",
        "purpose": "new car",
        "credit_amount": 2500,
        "savings_status": "<100",
        "employment": "4<=X<7",
        "installment_commitment": 2,
        "personal_status": "male mar/wid",
        "other_parties": "guarantor",
        "residence_since": 1,
        "property_magnitude": "no known property",
        "age": 50,
        "other_payment_plans": "none",
        "housing": "for free",
        "existing_credits": 1,
        "job": "high qualif/self emp/mgmt",
        "num_dependents": 2,
        "own_telephone": "none",
        "foreign_worker": "yes",
    }


def test_matches_feature_name_credit_risk() -> None:
    assert CreditRiskClassifierFG.match_feature_group_criteria("credit_risk", None) is True


def test_does_not_match_other_feature_names() -> None:
    assert CreditRiskClassifierFG.match_feature_group_criteria("something_else", None) is False
    assert CreditRiskClassifierFG.match_feature_group_criteria("credit_risk__attribution", None) is False


def test_input_features_cover_all_german_credit_columns() -> None:
    fg = CreditRiskClassifierFG()
    features = fg.input_features(None, None)
    names = {str(f.name) for f in features}
    for col in FEATURE_COLUMNS:
        assert col in names
    # customer_id is the join key (declared via index_columns), not an input feature.
    assert "customer_id" not in names


@pytest.mark.slow
def test_calculate_feature_predicts_credit_risk() -> None:
    df = pd.DataFrame([_fake_customer_row()])
    result = CreditRiskClassifierFG.calculate_feature(df, None)
    assert "credit_risk" in result
    assert len(result["credit_risk"]) == 1
    assert result["credit_risk"][0] in (0, 1)


@pytest.mark.slow
def test_artifact_roundtrip_reuses_model() -> None:
    df = pd.DataFrame([_fake_customer_row()])
    first = CreditRiskClassifierFG.calculate_feature(df, None)
    second = CreditRiskClassifierFG.calculate_feature(df, None)
    assert first == second
