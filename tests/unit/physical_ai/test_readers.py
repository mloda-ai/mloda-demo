from collections.abc import Callable

import numpy as np
from mloda.provider import FeatureSet
from mloda.user import Options

from mloda_demo.physical_ai.readers import DepthLog, DepthLogReader, DepthReaderA, DepthReaderB

FeatureSetFactory = Callable[..., FeatureSet]
MISMATCH = ("DepthToMetres", (("float32", "m"), ("uint16", "cm")))


def test_root_feature_group_reads_through_the_reader_family() -> None:
    assert isinstance(DepthLog.input_data(), DepthLogReader)


def test_reader_claims_log_columns_only() -> None:
    path = DepthReaderA.log_path()
    assert DepthReaderA.match_subclass_data_access(path, ["depth_raw", "frame"], Options()) == path
    assert DepthReaderA.match_subclass_data_access(path, ["depth_m"], Options()) is None
    assert DepthReaderA.match_subclass_data_access(None, ["depth_raw"], Options()) is None


def test_reader_declines_a_consumer_assuming_another_unit() -> None:
    options = Options(group={"assumed_units": MISMATCH})
    assert DepthReaderB.match_subclass_data_access(DepthReaderB.log_path(), ["depth_raw"], options) is None
    assert DepthReaderA.match_subclass_data_access(DepthReaderA.log_path(), ["depth_raw"], options) is not None


def test_logs_carry_each_device_encoding() -> None:
    for reader in (DepthReaderA, DepthReaderB):
        with np.load(reader.log_path()) as log:
            assert str(log["depth_raw"].dtype) == reader.encoding


def test_load_data_replays_frames_up_to_the_given_one(feature_set: FeatureSetFactory) -> None:
    full = DepthReaderA.load_data(DepthReaderA.log_path(), feature_set("depth_raw"))
    replay = DepthReaderA.load_data(DepthReaderA.log_path(), feature_set("depth_raw", replay_until=0))
    assert set(full["frame"]) == {0, 1, 2}
    assert set(replay["frame"]) == {0}


def test_readers_declare_unit_encoding_and_their_own_version() -> None:
    declared = DepthReaderB.declared_attributes(None)
    assert {key: declared[key] for key in ("unit", "encoding", "sensor", "frame")} == {
        "unit": "mm",
        "encoding": "uint16",
        "sensor": "device_b",
        "frame": "camera_optical",
    }
    assert declared["version"] != DepthReaderA.declared_attributes(None)["version"]
