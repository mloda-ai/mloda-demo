"""Readers: the only device-specific code. Each one declares what its integers mean, or does not."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from mloda.provider import INPUT_DATA_STAGE, BaseInputData, FeatureGroup, FeatureSet, record_match_rejection
from mloda.user import Options

from mloda_demo.physical_ai.clip import frame_index, load_depth
from mloda_demo.physical_ai.definition import PandasOnly

READER_COLUMNS = frozenset({"frame", "t_s", "source", "depth_raw", "depth_scale"})
REQUIRES_DECLARED_SCALE = "requires_declared_scale"


class DepthFrameReader(BaseInputData):
    """Reads a clip of 16-bit depth PNGs, one row per frame; a subclass fixes the scale."""

    scale: ClassVar[float | None] = None  # units per metre
    encoding: ClassVar[str] = "uint16"
    sensor: ClassVar[str] = ""
    dataset: ClassVar[str] = ""
    label: ClassVar[str] = ""

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if not isinstance(data_access, (str, Path)) or not READER_COLUMNS.issuperset(feature_names):
            return None
        consumer = options.get(REQUIRES_DECLARED_SCALE)
        if consumer is not None and cls.scale is None:
            record_match_rejection(
                cls.get_class_name(),
                f"{consumer} needs a declared scale; {cls.get_class_name()} delivers {cls.encoding} without one",
                stage=INPUT_DATA_STAGE,
            )
            return None
        return data_access

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        directory = Path(data_access)
        index = frame_index(directory)
        depth = [load_depth(directory / path) for path in index["depth"]]
        return pd.DataFrame(
            {
                "frame": index["frame"].to_numpy(),
                "t_s": index["t_s"].to_numpy(),
                "source": [f"{directory.name}/{path}" for path in index["depth"]],
                "depth_raw": pd.Series(depth, dtype=object),
                "depth_scale": np.full(len(index), np.nan if cls.scale is None else cls.scale),
            }
        )

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        version = hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()[:4]
        return {
            "scale": "undeclared" if cls.scale is None else f"{cls.scale:g} per m",
            "encoding": cls.encoding,
            "sensor": cls.sensor,
            "dataset": cls.dataset,
            "version": version,
        }


class DepthPng(DepthFrameReader):
    """Any 16-bit depth PNG. Says nothing about the unit."""

    sensor = "unknown"
    dataset = "unknown"
    label = "generic PNG"


class TumDepth(DepthFrameReader):
    """TUM RGB-D benchmark depth PNGs; 0 means no data."""

    scale = 5000  # per metre
    sensor = "Kinect on a Pioneer robot"
    dataset = "TUM RGB-D"
    label = "TUM"


class DepthFrames(PandasOnly, FeatureGroup):
    """The clip's frames; a group option keyed by the reader class picks the reader."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DepthFrameReader()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return DepthFrameReader().load(features)
