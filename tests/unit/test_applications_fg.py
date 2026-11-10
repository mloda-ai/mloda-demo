from mloda.provider import DataCreatorFeatureGroup

from mloda_demo.feature_groups.inputs.applications_fg import COLUMNS, ApplicationsFG


def test_is_data_creator_for_its_columns() -> None:
    assert issubclass(ApplicationsFG, DataCreatorFeatureGroup)
    assert ApplicationsFG.feature_names_supported() == COLUMNS


def test_calculate_feature_emits_expected_columns() -> None:
    df = ApplicationsFG.calculate_feature(None, None)
    assert set(df.columns) == COLUMNS
    assert len(df) == 5
    assert df["customer_id"].iloc[0].startswith("app-customer-")


def test_credit_amount_is_integer() -> None:
    df = ApplicationsFG.calculate_feature(None, None)
    assert df["credit_amount"].dtype.kind == "i"
