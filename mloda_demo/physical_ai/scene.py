"""Synthetic scene seen by both devices: no hardware, one known ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR
from mloda_demo.physical_ai.definition import CX

LOG_DIR = DEMO_DATA_DIR / "physical_ai"

TIMESTAMPS = (0.0, 0.2, 0.4)
# Object id -> (pixel column of its nearest point, nearest depth in metres per frame).
OBJECTS: dict[int, tuple[int, tuple[float, ...]]] = {
    7: (int(CX), (1.2, 1.2, 1.2)),
    3: (420, (6.0, 5.6, 5.2)),
    5: (520, (4.0, 4.0, 4.0)),
    2: (180, (8.5, 8.5, 8.5)),
    9: (260, (11.0, 11.0, 11.0)),
}
# Samples per object: (pixel offset, extra depth in metres) around its nearest point.
SAMPLES = ((0, 0.0), (-30, 0.05), (30, 0.05), (0, 0.1))


def depth_samples() -> pd.DataFrame:
    """One row per depth sample, depth in metres (float64)."""
    rows = [
        (frame, timestamp, object_id, u + du, depths[frame] + dd)
        for frame, timestamp in enumerate(TIMESTAMPS)
        for object_id, (u, depths) in OBJECTS.items()
        for du, dd in SAMPLES
    ]
    return pd.DataFrame(rows, columns=["frame", "timestamp", "object_id", "u", "depth"])


def write_logs(directory: Path = LOG_DIR) -> None:
    """Device A logs float32 metres, Device B uint16 millimetres (both valid under ROS REP 118)."""
    samples = depth_samples()
    common = {
        "frame": samples["frame"].to_numpy(np.int32),
        "timestamp": samples["timestamp"].to_numpy(np.float64),
        "object_id": samples["object_id"].to_numpy(np.int32),
        "u": samples["u"].to_numpy(np.int32),
    }
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "device_a.npz", **common, depth_raw=samples["depth"].to_numpy(np.float32))
    np.savez(directory / "device_b.npz", **common, depth_raw=np.rint(samples["depth"] * 1000).astype(np.uint16))


if __name__ == "__main__":
    write_logs()
