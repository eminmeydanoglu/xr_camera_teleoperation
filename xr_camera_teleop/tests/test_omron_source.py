import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xr_camera_teleop" / "camera"))

from camera_omron import LEVEL_STRAIGHT_DOWN, active_region, load_calibration  # noqa: E402
from xr_camera_teleop.teleop.camlevel import level_from_down, pitch_rotation  # noqa: E402

R_W_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # CONTRACT.md


def test_straight_down_level_matches_camlevel():
    R = np.asarray(LEVEL_STRAIGHT_DOWN)
    np.testing.assert_allclose(R, pitch_rotation(90), atol=1e-12)
    np.testing.assert_allclose(R, level_from_down([0, 0, 1]), atol=1e-12)


def test_overhead_axes_in_world():
    R_wc = R_W_CAM @ np.asarray(LEVEL_STRAIGHT_DOWN)
    np.testing.assert_allclose(R_wc @ [0, 0, 1], [0, 0, -1], atol=1e-12)   # optical axis: down
    np.testing.assert_allclose(R_wc @ [0, -1, 0], [1, 0, 0], atol=1e-12)   # image up: forward
    np.testing.assert_allclose(R_wc @ [1, 0, 0], [0, -1, 0], atol=1e-12)   # image right: operator right


def test_active_region_matches_receiver():
    assert active_region((1224, 1024)) == (210, 0, 860, 720)   # full field of view, side bars
    assert active_region((1224, 688)) == (0, 1, 1280, 718)     # wide mode: 16:9 band, 1 px bars


def test_calibration_resolution_checked(tmp_path):
    with pytest.raises(RuntimeError, match="not calibrated"):
        load_calibration(860, 720, tmp_path / "missing.json")
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"fx": 700, "fy": 700, "cx": 430, "cy": 360, "width": 860, "height": 720,
                                "distortion": [0] * 5, "source": "test"}))
    assert load_calibration(860, 720, path)["fx"] == 700
    with pytest.raises(RuntimeError, match="recalibrate"):
        load_calibration(1280, 720, path)
