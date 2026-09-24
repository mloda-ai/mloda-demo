"""The robot clip (TUM RGB-D, freiburg2_pioneer_slam) and the corridor the nearest reading is taken from."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from mloda_demo.feature_groups.inputs.paths import DEMO_DATA_DIR

CLIP_DIR = DEMO_DATA_DIR / "physical_ai" / "pioneer_slam"
CORRIDOR = "columns 1/3 to 2/3, rows 1/4 to 3/4"
PERCENTILE = 2.0


def frame_index(directory: Path = CLIP_DIR) -> pd.DataFrame:
    """One row per frame: frame, t_s, and the depth and rgb paths relative to the directory."""
    return pd.read_csv(directory / "frames.csv")


def load_depth(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path), dtype=np.uint16)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def corridor(image: np.ndarray) -> np.ndarray:
    rows, columns = image.shape
    return image[rows // 4 : 3 * rows // 4, columns // 3 : 2 * columns // 3]


def nearest_in_corridor(values: np.ndarray) -> float:
    """The 2nd percentile of the valid (finite, positive) corridor values; NaN when nothing is valid."""
    band = corridor(values).astype(np.float64)
    valid = band[np.isfinite(band) & (band > 0)]
    return float(np.percentile(valid, PERCENTILE)) if valid.size else float("nan")
