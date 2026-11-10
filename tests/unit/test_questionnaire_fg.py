from mloda.provider import DataCreatorFeatureGroup

from mloda_demo.feature_groups.inputs.questionnaire_fg import CATEGORICAL_FIELDS, COLUMNS, QuestionnaireFG


def test_is_data_creator_for_its_columns() -> None:
    assert issubclass(QuestionnaireFG, DataCreatorFeatureGroup)
    assert QuestionnaireFG.feature_names_supported() == COLUMNS


def test_calculate_feature_populates_all_categorical_fields() -> None:
    df = QuestionnaireFG.calculate_feature(None, None)
    assert len(df) == 5
    for field in CATEGORICAL_FIELDS:
        assert field in df.columns
        assert df[field].notna().all()
        assert (df[field] != "").all()


def test_customer_id_present() -> None:
    df = QuestionnaireFG.calculate_feature(None, None)
    assert df["customer_id"].iloc[0].startswith("app-customer-")
