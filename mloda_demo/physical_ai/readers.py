"""Device readers: the only device-specific code."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from mloda.provider import INPUT_DATA_STAGE, BaseInputData, FeatureGroup, FeatureSet, record_match_rejection
from mloda.user import Options

from mloda_demo.physical_ai.definition import PandasOnly
from mloda_demo.physical_ai.scene import LOG_DIR

LOG_COLUMNS = frozenset({"frame", "timestamp", "object_id", "u", "depth_raw"})


class DepthLogReader(BaseInputData):
    """Reads a synthetic depth log; each subclass fixes one device's encoding and unit."""

    log_file: ClassVar[str] = ""
    encoding: ClassVar[str] = ""
    unit: ClassVar[str] = ""
    sensor: ClassVar[str] = ""
    label: ClassVar[str] = ""

    @classmethod
    def log_path(cls) -> str:
        return str(LOG_DIR / cls.log_file)

    @classmethod
    def match_subclass_data_access(cls, data_access: Any, feature_names: list[str], options: Options) -> Any:
        if not isinstance(data_access, str) or not LOG_COLUMNS.issuperset(feature_names):
            return None
        assumed = options.get("assumed_units")
        if assumed is not None and dict(assumed).get(cls.encoding) != cls.unit:
            record_match_rejection(
                cls.get_class_name(),
                f"DepthToMetres assumes {cls.encoding} depth in {dict(assumed).get(cls.encoding)}, "
                f"but {cls.get_class_name()} delivers {cls.unit}",
                stage=INPUT_DATA_STAGE,
            )
            return None
        return data_access

    @classmethod
    def load_data(cls, data_access: Any, features: FeatureSet) -> Any:
        replay_until = features.get_options_key("replay_until")
        with np.load(data_access) as log:
            data = pd.DataFrame({name: log[name] for name in log.files})
        if replay_until is not None:
            data = data[data["frame"] <= replay_until].reset_index(drop=True)
        return data

    @classmethod
    def declared_attributes(cls, features: FeatureSet | None) -> dict[str, str]:
        return {"unit": cls.unit, "encoding": cls.encoding, "sensor": cls.sensor, "frame": "camera_optical"}


class DepthReaderA(DepthLogReader):
    """Device A: depth as float32 metres."""

    log_file = "device_a.npz"
    encoding = "float32"
    unit = "m"
    sensor = "device_a"
    label = "device A"


class DepthReaderB(DepthLogReader):
    """Device B: depth as uint16 millimetres."""

    log_file = "device_b.npz"
    encoding = "uint16"
    unit = "mm"
    sensor = "device_b"
    label = "device B"


class DepthLog(PandasOnly, FeatureGroup):
    """Depth log columns; a group option keyed by the reader class picks the device."""

    @classmethod
    def input_data(cls) -> BaseInputData | None:
        return DepthLogReader()

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return DepthLogReader().load(features)
