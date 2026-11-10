"""Readers: the only device-specific code. Each one declares what its integers mean, or does not."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from mloda.provider import ClaimRoute, FeatureSet, NamePolicy, SourceFeatureGroup, SourceMatch
from mloda.user import DataAccessCollection, Options

from mloda_demo.pandas_only import PandasOnly
from mloda_demo.physical_ai.clip import frame_index, load_depth

READER_COLUMNS = frozenset({"frame", "t_s", "source", "depth_raw", "depth_scale"})


class DepthFrameReader(PandasOnly, SourceFeatureGroup):
    """Reads a clip of 16-bit depth PNGs, one row per frame; a subclass fixes the scale.

    A group option keyed by the reader class picks it: `Options(group={TumDepth: clip_dir})`.
    """

    CLAIM_ROUTES = (ClaimRoute("clip", NamePolicy.DECLARED, False),)

    scale: ClassVar[float | None] = None  # units per metre
    encoding: ClassVar[str] = "uint16"
    sensor: ClassVar[str] = ""
    dataset: ClassVar[str] = ""
    label: ClassVar[str] = ""

    @classmethod
    def declared_names(cls) -> frozenset[str]:
        return READER_COLUMNS

    @classmethod
    def find_sources(
        cls, route: ClaimRoute, feature_name: str, options: Options, data_access_collection: DataAccessCollection | None
    ) -> list[SourceMatch]:
        clip = next((options.get(key) for key in cls.pointer_keys() if key in options), None)
        if not isinstance(clip, (str, Path)) or not (Path(clip) / "frames.csv").is_file():
            return []
        return [SourceMatch(source=str(clip), access=Path(clip))]

    @classmethod
    def load_neutral(cls, match: SourceMatch, features: Any) -> Any:
        directory = Path(match.access).resolve()
        index = frame_index(directory)
        paths = [(directory / path).resolve() for path in index["depth"]]
        if any(directory not in path.parents for path in paths):
            raise ValueError("frames.csv points outside the clip directory")
        depth = [load_depth(path) for path in paths]
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
        """No scale key when the scale is unknown, so a consumer requiring one refuses this reader."""
        version = hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()[:4]
        scale = {} if cls.scale is None else {"scale": f"{cls.scale:g} per m"}
        return {**scale, "encoding": cls.encoding, "sensor": cls.sensor, "dataset": cls.dataset, "version": version}


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
