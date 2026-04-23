from mloda.provider import DataCreator

from mloda_demo.feature_groups.inputs.applications_fg import COLUMNS, ApplicationsFG


def test_input_data_is_data_creator() -> None:
    input_data = ApplicationsFG.input_data()
    assert isinstance(input_data, DataCreator)


def test_calculate_feature_emits_expected_columns() -> None:
    df = ApplicationsFG.calculate_feature(None, None)
    assert set(df.columns) == COLUMNS
    assert len(df) == 5
    assert df["customer_id"].iloc[0].startswith("app-customer-")


def test_credit_amount_is_integer() -> None:
    df = ApplicationsFG.calculate_feature(None, None)
    assert df["credit_amount"].dtype.kind == "i"
