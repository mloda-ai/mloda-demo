"""The robot chain: three depth sources in three formats, and one definition of when to stop."""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureChainParserMixin, FeatureGroup, FeatureSet
from mloda.user import Feature, FeatureName, Options
from numpy.typing import NDArray

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR
from mloda_demo.one_process.chain import PandasOnly, source_of
from mloda_demo.physical_ai.clip import CLIP_DIR, frame_index, load_depth, nearest_in_corridor

REDWOOD_DIR = DEMO_DATA_DIR / "physical_ai" / "redwood" / "depth"
STOP_BELOW_M = 1.0
Frames = tuple[NDArray[Any], ...]


@cache
def tum_frames() -> Frames:
    """The TUM Pioneer clip: 16-bit PNGs, 5000 per metre, 0 means no data."""
    return tuple(load_depth(CLIP_DIR / path) for path in frame_index(CLIP_DIR)["depth"])


@cache
def redwood_frames() -> Frames:
    """Five frames of a rendered living room: 16-bit PNGs in millimetres."""
    return tuple(load_depth(path) for path in sorted(REDWOOD_DIR.glob("*.png")))


@cache
def sim_frames() -> Frames:
    """A flat wall approaching from 3 m to 0.4 m over ten frames, as float metres."""
    return tuple(np.full((480, 640), metres, dtype=np.float32) for metres in np.linspace(3.0, 0.4, 10))


def source_table(name: str, frames: Frames, scale: float) -> pd.DataFrame:
    """One row per frame: the raw depth and the source's declared units per metre."""
    return pd.DataFrame({name: pd.Series(list(frames), dtype=object), f"{name}_scale": scale})


class Tum(PandasOnly, FeatureGroup):
    """The robot's log: TUM RGB-D benchmark PNGs, 5000 per metre."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"tum", "tum_scale"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return source_table("tum", tum_frames(), 5000)


class Redwood(PandasOnly, FeatureGroup):
    """A simulator's frames: rendered scene, PNGs in millimetres."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"redwood", "redwood_scale"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return source_table("redwood", redwood_frames(), 1000)


class Sim(PandasOnly, FeatureGroup):
    """A generated wall, delivered as float metres."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DataCreator({"sim", "sim_scale"})

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return source_table("sim", sim_frames(), 1)


def to_metres(raw: NDArray[Any], scale: float) -> NDArray[np.float64]:
    """Raw depth over the declared scale; 0 stays invalid as NaN."""
    metres = raw.astype(np.float64) / scale
    metres[raw == 0] = np.nan
    return metres


class Metres(PandasOnly, FeatureChainParserMixin, FeatureGroup):
    """Depth in metres, whatever the source delivered."""

    PREFIX_PATTERN = r"^.+__metres$"

    def input_features(self, options: Options, feature_name: FeatureName) -> set[Feature] | None:
        source = source_of(feature_name)
        return {Feature(source), Feature(f"{source}_scale")}

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            source = source_of(feature.name)
            metres = [to_metres(raw, scale) for raw, scale in zip(data[source], data[f"{source}_scale"], strict=True)]
            data[str(feature.name)] = pd.Series(metres, index=data.index, dtype=object)
        return data


class Nearest(PandasOnly, FeatureChainParserMixin, FeatureGroup):
    """The nearest valid reading in the central corridor, per frame."""

    PREFIX_PATTERN = r"^.+__nearest$"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[str(feature.name)] = [nearest_in_corridor(metres) for metres in data[source_of(feature.name)]]
        return data


class Stop(PandasOnly, FeatureChainParserMixin, FeatureGroup):
    """Stop below one metre, or when there is no reading."""

    PREFIX_PATTERN = r"^.+__stop$"

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        for feature in features.features:
            data[str(feature.name)] = ~(data[source_of(feature.name)] >= STOP_BELOW_M)
        return data
