import inspect

from mloda_demo.retail.kpis import NetSpend30d
from mloda_demo.retail.runs import definition


def test_the_slide_lists_the_shared_hooks_by_name_with_the_inputs_and_the_math_in_full() -> None:
    shared, per_use_case = (part.splitlines() for part in definition(NetSpend30d))
    assert shared[0] == "class NetSpend30d(Kpi, FeatureGroup):"
    assert shared[1].strip().startswith('"""Net spend, 30 days')
    assert [line.strip() for line in shared if "def " in line] == [
        "def input_features(...)",
        "def return_data_type_rule(...)",
        "def validate_output_features(...)",
    ]
    assert 'Feature("customer_id", options={"needs": "cancellations"}, feature_group="Orders"),' in [
        line.strip() for line in shared
    ]
    assert not any("OWNER" in line for line in shared), "no owner concept in mloda"
    assert per_use_case[0].strip().startswith("def calculate_feature(cls")
    source = inspect.getsource(NetSpend30d).splitlines()
    assert all(line in source for line in per_use_case), "the math is shown as written"
    assert all(line in source for line in shared if "def " not in line), "abridged, never rewritten"
