import numpy as np

from xr_camera_teleop.teleop import mano_to_webxr as m
from xr_camera_teleop.teleop.synthetic_stream import canonical_hand_mano21


def test_shape_and_order():
    kp = np.random.default_rng(0).normal(size=(21, 3))
    w = m.mano21_to_webxr25(kp)
    assert w.shape == (25, 3)
    np.testing.assert_allclose(w[0], kp[0])
    np.testing.assert_allclose(w[1:5], kp[1:5])           # thumb direct
    for f in range(4):                                     # index..pinky
        base = 5 + 5 * f
        mcp = kp[5 + 4 * f]
        np.testing.assert_allclose(w[base], kp[0] + 0.3 * (mcp - kp[0]))   # metacarpal
        np.testing.assert_allclose(w[base + 1:base + 5], kp[5 + 4 * f:9 + 4 * f])
    # tips used by Dex3 DexPilot indices 4, 9, 14
    np.testing.assert_allclose(w[4], kp[4])
    np.testing.assert_allclose(w[9], kp[8])
    np.testing.assert_allclose(w[14], kp[12])


def test_metacarpal_between_wrist_and_mcp():
    kp = canonical_hand_mano21("left")
    w = m.mano21_to_webxr25(kp)
    for f in range(4):
        d = np.linalg.norm(w[5 + 5 * f] - w[0]) / np.linalg.norm(w[5 + 5 * f + 1] - w[0])
        assert abs(d - 0.3) < 1e-9


def test_wrist_frame_openxr_convention():
    # left hand palm-down fingers forward in a z-up frame: x fwd, y left(pinky side), z up
    kp = canonical_hand_mano21("left")        # local XR wrist frame already
    R, o = m.keypoint_wrist_frame(kp, "left")
    np.testing.assert_allclose(R, np.eye(3), atol=0.15)
    assert np.isclose(np.linalg.det(R), 1)
    kr = canonical_hand_mano21("right")
    Rr, _ = m.keypoint_wrist_frame(kr, "right")
    np.testing.assert_allclose(Rr, np.eye(3), atol=0.15)   # right: y = pinky->index
    # index->pinky axis for left is +y, pinky->index for right is +y
    assert kp[17][1] > kp[5][1]
    assert kr[5][1] > kr[17][1]


def test_wrist_frame_rotates_with_hand():
    rng = np.random.default_rng(1)
    for side in ("left", "right"):
        kp = canonical_hand_mano21(side, grip=0.4)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = rng.normal(size=3)
        R0, _ = m.keypoint_wrist_frame(kp, side)
        R1, o1 = m.keypoint_wrist_frame(kp @ Q.T + t, side)
        np.testing.assert_allclose(R1, Q @ R0, atol=1e-9)
        np.testing.assert_allclose(o1, t, atol=1e-12)


def test_compare_with_wilor_rot_constant_offset():
    kp = canonical_hand_mano21("right")
    C = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0.0]])
    Rk, _ = m.keypoint_wrist_frame(kp, "right")
    _, C2, ang = m.compare_with_wilor_rot(kp, Rk @ C, "right")
    np.testing.assert_allclose(C2, C, atol=1e-9)


def test_pinch_and_squeeze():
    o = m.mano21_to_webxr25(canonical_hand_mano21("left", 0.0, 0.0))
    c = m.mano21_to_webxr25(canonical_hand_mano21("left", 1.0, 0.0))
    p = m.mano21_to_webxr25(canonical_hand_mano21("left", 0.0, 1.0))
    assert m.squeeze_value(o) < 0.15 and m.squeeze_value(c) > 0.8
    assert m.pinch_distance(p) < 0.02 < m.pinch_distance(o)
