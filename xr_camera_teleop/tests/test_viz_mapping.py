import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from xr_camera_teleop.viz.mapping import (ANCHOR_DZ, ANCHOR_XY, BONES, R_W_CAM, HandMapper, HandSmoother,  # noqa: E402
                                  cam_to_world_points, project_so3)
from xr_camera_teleop.viz.synth import local_hand, make_message  # noqa: E402


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
            "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def test_axes():
    f = lambda v: cam_to_world_points(v, cam_height=0.0)  # noqa: E731
    assert np.allclose(f([0, 0, 1]), [1, 0, 0])   # forward -> +x
    assert np.allclose(f([1, 0, 0]), [0, -1, 0])  # image right -> -y
    assert np.allclose(f([0, 1, 0]), [0, 0, -1])  # image down -> -z
    assert np.isclose(np.linalg.det(R_W_CAM), 1)
    assert np.allclose(cam_to_world_points([0, 0, 0], 1.5), [0, 0, 1.5])


def test_left_hand_is_plus_y():
    m = make_message(0, 0.0)
    pl = cam_to_world_points(m["hands"]["left"]["wrist_pos"])[1]
    pr = cam_to_world_points(m["hands"]["right"]["wrist_pos"])[1]
    assert pl > 0 > pr


def test_synth_hand_curl_changes_tip_distance():
    k0, k1 = local_hand("right", 0.0), local_hand("right", 1.0)
    assert np.linalg.norm(k1[8] - k1[0]) < 0.7 * np.linalg.norm(k0[8] - k0[0])
    assert len(BONES) == 20


def _pose(t, R=None):
    return np.array(t, float), np.eye(3) if R is None else R


def test_relative_no_jump_and_motion():
    mp = HandMapper("relative", scale=0.8, cam_height=1.5)
    msg = make_message(0, 0.0)["hands"]["left"]
    p, R, k = (np.array(msg["wrist_pos"]), np.array(msg["wrist_rot"]), np.array(msg["keypoints"]))
    assert mp.map("left", p, R, k) is None  # uncalibrated
    mp.calibrate({"left": (p, R)})
    p0, R0, k0 = mp.map("left", p, R, k)
    ax, ay = ANCHOR_XY["left"]
    assert np.allclose(p0, [ax, ay, 1.5 + ANCHOR_DZ])
    assert np.allclose(R0, R_W_CAM @ R)  # no orientation jump
    # fingers keep wrist-local shape
    kw = k @ R_W_CAM.T
    assert np.allclose(k0 - p0, kw - R_W_CAM @ p, atol=1e-9)
    # move hand 0.1 m camera-right -> robot -y by 0.08
    p1, R1, k1 = mp.map("left", p + [0.1, 0, 0], R, k + [0.1, 0, 0])
    assert np.allclose(p1 - p0, [0, -0.08, 0])
    assert np.allclose(k1 - p1, k0 - p0)


def test_relative_rotation_delta_in_world():
    mp = HandMapper("relative", scale=1.0, neutral_rot="identity")
    p, R = np.array([0.0, 0, 0.5]), np.eye(3)
    kps = np.zeros((21, 3)) + p
    mp.calibrate({"right": (p, R)})
    Rd_w = rot("z", 0.3)  # delta rotation in W
    R_cam = R_W_CAM.T @ Rd_w @ R_W_CAM  # same rotation expressed in CAM
    _, Rn, _ = mp.map("right", p, R_cam, kps)
    assert np.allclose(Rn @ Rn.T, np.eye(3)) and np.isclose(np.linalg.det(Rn), 1)
    assert np.allclose(Rn, Rd_w, atol=1e-9)


def test_absolute_scale_about_camera():
    mp = HandMapper("absolute", scale=2.0, cam_height=1.0)
    p, R, k = mp.map("left", [0, 0, 0.5], np.eye(3), np.zeros((21, 3)) + [0, 0, 0.5])
    assert np.allclose(p, [1.0, 0, 1.0])


def test_smoother_passthrough_and_so3():
    s = HandSmoother()
    msg = make_message(0, 0.0)["hands"]["right"]
    args = (np.array(msg["wrist_pos"]), np.array(msg["wrist_rot"]), np.array(msg["keypoints"]))
    p, R, k = s(*args, 0.0)
    assert np.allclose(k, args[2])
    p, R, k = s(*args, 0.03)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(project_so3(rot("x", 0.2) * 1.01), rot("x", 0.2))


def test_robot_anchor_relative_matches_clutch_neutral():
    from xr_camera_teleop.viz.mapping import HandMapper, robot_anchors
    m = HandMapper("relative", None, 1.5, neutral_rot="identity", anchors=robot_anchors(1.0))
    m.calibrate({"left": ([0, 0, 0.5], np.eye(3))})
    p, R, _ = m.map("left", [0, 0, 0.5], np.eye(3), np.zeros((21, 3)))
    assert np.allclose(p, [0.2498, 0.1487, 1.0952]) and np.allclose(R, np.eye(3))
    p, _, _ = m.map("left", [0, 0, 0.6], np.eye(3), np.zeros((21, 3)))  # cam forward 10 cm * k=0.8
    assert np.allclose(p[0], 0.2498 + 0.08)


def test_extrapolator_reduces_lag():
    from xr_camera_teleop.viz.mapping import Extrapolator
    e = Extrapolator()
    for t in (0.0, 1 / 30):
        e.update(t, [t, 0, 0], np.eye(3), np.zeros((21, 3)))
    p, R, _ = e.at(1 / 30 + 1 / 60)
    assert abs(p[0] - (1 / 30 + 1 / 60)) < 1e-9 and np.allclose(R, np.eye(3))
