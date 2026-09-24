from collections.abc import Callable

import numpy as np
from mloda.provider import FeatureSet
from mloda.user import Options

from mloda_demo.physical_ai.clip import CLIP_DIR, frame_index, load_rgb
from mloda_demo.physical_ai.readers import REQUIRES_DECLARED_SCALE, DepthFrameReader, DepthFrames, DepthPng, TumDepth

FeatureSetFactory = Callable[..., FeatureSet]
CLIP = str(CLIP_DIR)


def test_root_feature_group_reads_through_the_reader_family() -> None:
    assert isinstance(DepthFrames.input_data(), DepthFrameReader)


def test_reader_claims_clip_columns_only() -> None:
    assert DepthPng.match_subclass_data_access(CLIP, ["depth_raw", "frame"], Options()) == CLIP
    assert DepthPng.match_subclass_data_access(CLIP, ["depth_m"], Options()) is None
    assert DepthPng.match_subclass_data_access(None, ["depth_raw"], Options()) is None


def test_undeclared_reader_declines_a_consumer_that_needs_a_scale() -> None:
    options = Options(group={REQUIRES_DECLARED_SCALE: "DepthToMetres"})
    assert DepthPng.match_subclass_data_access(CLIP, ["depth_raw"], options) is None
    assert TumDepth.match_subclass_data_access(CLIP, ["depth_raw"], options) == CLIP


def test_load_data_gives_one_row_per_frame_with_the_declared_scale(feature_set: FeatureSetFactory) -> None:
    generic = DepthPng.load_data(CLIP, feature_set("depth_raw"))
    tum = TumDepth.load_data(CLIP, feature_set("depth_raw"))
    assert len(generic) == len(tum) == 54
    assert generic["depth_raw"].iloc[0].dtype == np.uint16
    assert generic["depth_raw"].iloc[0].shape == (480, 640)
    assert np.isnan(generic["depth_scale"]).all()
    assert (tum["depth_scale"] == 5000.0).all()
    assert generic["source"].iloc[53] == "pioneer_slam/depth/053.png"


def test_clip_index_pairs_each_depth_frame_with_its_camera_image() -> None:
    frames = frame_index(CLIP_DIR)
    assert list(frames["frame"]) == list(range(54))
    rgb = load_rgb(CLIP_DIR / frames.loc[frames["frame"] == 53, "rgb"].iloc[0])
    assert (rgb.shape, rgb.dtype) == ((480, 640, 3), np.uint8)


def test_readers_declare_scale_encoding_and_their_own_version() -> None:
    declared = TumDepth.declared_attributes(None)
    assert {key: declared[key] for key in ("scale", "encoding", "dataset")} == {
        "scale": "5000 per m",
        "encoding": "uint16",
        "dataset": "TUM RGB-D",
    }
    assert DepthPng.declared_attributes(None)["scale"] == "undeclared"
    assert declared["version"] != DepthPng.declared_attributes(None)["version"]
