import pandas as pd
import pytest

from mloda_demo.one_process.lineage import Lineage, Step, label, rows


def _lineage(features: list[str], steps: list[Step]) -> Lineage:
    return Lineage(tuple(features), {name: pd.DataFrame({name: [1.0]}) for name in features}, tuple(steps))


def test_label_reads_like_the_feature_name():
    assert label("mloda_demo.one_process.ml.MonthlyPayment") == "monthly_payment"
    assert label("Tum") == "tum"
    assert rows(1) == "1 row"
    assert rows(54) == "54 rows"


def test_welded_chains_share_nothing():
    steps = [
        Step("TumFrames", (), ("tum_depth",)),
        Step("RedwoodFrames", (), ("redwood_depth",)),
        Step("TumStop", ("tum_depth",), ("tum_stop",)),
        Step("RedwoodStop", ("redwood_depth",), ("redwood_stop",)),
    ]
    assert _lineage(["tum_stop", "redwood_stop"], steps).chains() == [
        ["tum_frames", "tum_stop"],
        ["redwood_frames", "redwood_stop"],
    ]


def test_two_sources_merge_into_one_definition():
    steps = [
        Step("Tum", (), ("tum", "tum_scale")),
        Step("Redwood", (), ("redwood", "redwood_scale")),
        Step("Metres", ("redwood", "redwood_scale"), ("redwood__metres",)),
        Step("Metres", ("tum", "tum_scale"), ("tum__metres",)),
    ]
    assert _lineage(["tum__metres", "redwood__metres"], steps).chains() == [["tum", "metres"], ["redwood", "metres"]]


def test_a_step_reading_its_source_and_an_earlier_column_stays_one_chain():
    steps = [
        Step("Finance", (), ("finance_customer", "finance_gross")),
        Step("Revenue", ("finance_gross",), ("finance__revenue",)),
        Step("PerCustomer", ("finance__revenue", "finance_customer"), ("finance__revenue__per_customer",)),
    ]
    assert _lineage(["finance__revenue__per_customer"], steps).chain("finance__revenue__per_customer") == [
        "finance",
        "revenue",
        "per_customer",
    ]


def test_a_step_fed_by_two_chains_is_refused():
    steps = [Step("A", (), ("a",)), Step("B", (), ("b",)), Step("Join", ("a", "b"), ("joined",))]
    with pytest.raises(ValueError, match="several steps"):
        _lineage(["joined"], steps).chains()


def test_html_holds_the_picture_and_the_row_counts():
    steps = [Step("Tum", (), ("tum",)), Step("Metres", ("tum",), ("tum__metres",))]
    html = _lineage(["tum__metres"], steps).html()
    assert "<svg" in html
    assert 'width="48%"' in html
    assert "tum__metres 1 row" in html
