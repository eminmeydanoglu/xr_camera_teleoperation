import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
mujoco = pytest.importorskip("mujoco")
from xr_camera_teleop.viz.hands_scene import HandsScene  # noqa: E402
from xr_camera_teleop.viz.synth import make_message  # noqa: E402
from xr_camera_teleop.viz.mapping import HandMapper  # noqa: E402


def test_scene_poses_keypoints():
    sc = HandsScene(1.5)
    h = make_message(0, 0.0)["hands"]["left"]
    res = HandMapper("absolute").map("left", h["wrist_pos"], h["wrist_rot"], h["keypoints"])
    sc.set_state("left", "live")
    sc.pose("left", *res)
    sc.forward()
    body = sc.model.body("left_j8")
    assert np.allclose(sc.data.xpos[body.id], res[2][8], atol=1e-9)
    sc.set_state("left", "hidden")
    assert sc.data.mocap_pos[sc.mocap["left"][0]][2] < -10
