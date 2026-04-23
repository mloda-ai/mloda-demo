from mloda.provider import DataCreator

from mloda_demo.feature_groups.inputs.financials_fg import COLUMNS, FinancialsFG


def test_input_data_is_data_creator() -> None:
    assert isinstance(FinancialsFG.input_data(), DataCreator)


def test_calculate_feature_emits_one_row_per_customer() -> None:
    df = FinancialsFG.calculate_feature(None, None)
    assert set(df.columns) == COLUMNS
    assert len(df) == 5


def test_numeric_fields_are_integers() -> None:
    df = FinancialsFG.calculate_feature(None, None)
    for col in ("duration", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents"):
        assert df[col].dtype.kind == "i", f"{col} should be integer"
