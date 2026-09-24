import pytest

from mloda_demo.one_process import mloda_run

ROBOT = ["tum", "metres", "nearest", "stop"]


def test_welded_pipelines_share_no_step():
    lineage = mloda_run(["tum_stop", "redwood_stop"])
    assert lineage.chains() == [
        ["tum_frames", "tum_metres", "tum_nearest", "tum_stop"],
        ["redwood_frames", "redwood_metres", "redwood_nearest", "redwood_stop"],
    ]
    assert len(lineage.tables["tum_stop"]) == 54
    assert len(lineage.tables["redwood_stop"]) == 5


def test_two_readers_feed_one_definition():
    lineage = mloda_run(["tum__metres__nearest__stop", "redwood__metres__nearest__stop"])
    assert lineage.chains() == [ROBOT, ["redwood", *ROBOT[1:]]]
    tum = lineage.tables["tum__metres__nearest__stop"]["tum__metres__nearest__stop"]
    assert tum.sum() == 8  # the chair: frames 46 to 53
    assert tum.iloc[-1]
    assert not lineage.tables["redwood__metres__nearest__stop"]["redwood__metres__nearest__stop"].any()


def test_the_welded_copy_and_the_shared_definition_agree():
    welded = mloda_run(["tum_stop"]).tables["tum_stop"]["tum_stop"]
    shared = mloda_run(["tum__metres__nearest__stop"]).tables["tum__metres__nearest__stop"]
    assert welded.tolist() == shared["tum__metres__nearest__stop"].tolist()


def test_offline_and_online_get_the_same_monthly_payment():
    lineage = mloda_run(["history__monthly_payment", "applicant__monthly_payment"])
    assert lineage.chains() == [["history", "monthly_payment"], ["applicant", "monthly_payment"]]
    history = lineage.tables["history__monthly_payment"]["history__monthly_payment"]
    assert len(history) == 1000
    assert history.iloc[0] == pytest.approx(1169 / 6)
    applicant = lineage.tables["applicant__monthly_payment"]["applicant__monthly_payment"]
    assert applicant.tolist() == pytest.approx([4200 / 36])


def test_two_departments_share_one_revenue():
    lineage = mloda_run(["finance__revenue__per_customer", "sales__revenue__per_customer"])
    assert lineage.chains() == [["finance", "revenue", "per_customer"], ["sales", "revenue", "per_customer"]]
    finance = lineage.tables["finance__revenue__per_customer"]["finance__revenue__per_customer"]
    assert finance.tolist() == [2460, 1320, 2460, 4200, 1320, 2460, 4200, 700]
    sales = lineage.tables["sales__revenue__per_customer"]["sales__revenue__per_customer"]
    assert sales.tolist() == [2430, 5130, 1500, 2430, 900, 5130]


def test_a_synthetic_source_joins_the_same_chain():
    lineage = mloda_run(["tum__metres__nearest__stop", "redwood__metres__nearest__stop", "sim__metres__nearest__stop"])
    assert lineage.chains() == [ROBOT, ["redwood", *ROBOT[1:]], ["sim", *ROBOT[1:]]]
    sim = lineage.tables["sim__metres__nearest__stop"]["sim__metres__nearest__stop"]
    assert sim.tolist() == [False] * 7 + [True] * 3
    assert 'width="96%"' in lineage.html()
