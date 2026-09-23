from pathlib import Path

import numpy as np

from mloda_demo.physical_ai.definition import CX
from mloda_demo.physical_ai.scene import LOG_DIR, depth_samples, write_logs


def test_committed_logs_match_the_generator(tmp_path: Path) -> None:
    write_logs(tmp_path)
    for name in ("device_a.npz", "device_b.npz"):
        with np.load(LOG_DIR / name) as committed, np.load(tmp_path / name) as generated:
            assert committed.files == generated.files
            for key in committed.files:
                assert committed[key].dtype == generated[key].dtype
                np.testing.assert_array_equal(committed[key], generated[key])


def test_object_7_nearest_sample_is_straight_ahead_at_1_2_m() -> None:
    samples = depth_samples()
    object_7 = samples[samples["object_id"] == 7]
    nearest = object_7.loc[object_7["depth"].idxmin()]
    assert nearest["u"] == CX
    assert nearest["depth"] == 1.2
    assert set(object_7["frame"]) == {0, 1, 2}
