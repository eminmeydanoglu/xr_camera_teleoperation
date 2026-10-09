import numpy as np

from xr_camera_teleop.teleop import filters as F
from xr_camera_teleop.teleop.clutch import HandClutch, neutral_from_tv_wrapper_const


def rz(a):
    return F.rot_exp(np.array([0, 0, a]))


def rx(a):
    return F.rot_exp(np.array([a, 0, 0]))


def neutral():
    T = np.eye(4)
    T[:3, 3] = [0.25, 0.15, 0.09]
    return T


def test_uncalibrated_outputs_neutral():
    c = HandClutch(neutral())
    T, info = c.update(0.0, np.array([5, 5, 5.0]), rx(1.0), True)
    np.testing.assert_allclose(T, neutral())
    assert not info["calibrated"]


def test_no_jump_at_calibration_any_human_pose():
    c = HandClutch(neutral())
    p, R = np.array([0.4, -0.3, 0.2]), rx(0.7) @ rz(2.1)
    before, _ = c.update(0.0, p, R, True)
    c.request_calibration()
    after, info = c.update(0.033, p, R, True)
    assert info["calibrated"]
    np.testing.assert_allclose(after, neutral(), atol=1e-12)
    np.testing.assert_allclose(before, neutral(), atol=1e-12)


def test_translation_scaled_by_k():
    c = HandClutch(neutral(), k=0.8)
    p0, R0 = np.array([0.4, 0.1, 0.0]), np.eye(3)
    c.request_calibration()
    c.update(0, p0, R0, True)
    T, _ = c.update(0.1, p0 + [0.1, 0.05, -0.02], R0, True)
    np.testing.assert_allclose(T[:3, 3], neutral()[:3, 3] + 0.8 * np.array([0.1, 0.05, -0.02]))
    c2 = HandClutch(neutral(), k=[1.0, 0.5, 2.0])
    c2.request_calibration()
    c2.update(0, p0, R0, True)
    T, _ = c2.update(0.1, p0 + [0.1, 0.1, 0.01], R0, True)
    np.testing.assert_allclose(T[:3, 3] - neutral()[:3, 3], [0.1, 0.05, 0.02])


def test_rotation_delta_applied_in_world():
    Rn = rx(0.5)
    Tn = np.eye(4)
    Tn[:3, :3] = Rn
    c = HandClutch(Tn)
    Rh0 = rx(1.0) @ rz(0.3)
    c.request_calibration()
    c.update(0, np.zeros(3), Rh0, True)
    D = rz(0.4)                              # operator yaws about WORLD z by 0.4
    T, _ = c.update(0.1, np.zeros(3), D @ Rh0, True)
    np.testing.assert_allclose(T[:3, :3], D @ Rn, atol=1e-12)   # robot yaws about world z too
    # independence of the human body-frame convention: R_h -> R_h @ C
    C = rx(0.9) @ rz(-0.4)
    c2 = HandClutch(Tn)
    c2.request_calibration()
    c2.update(0, np.zeros(3), Rh0 @ C, True)
    T2, _ = c2.update(0.1, np.zeros(3), D @ Rh0 @ C, True)
    np.testing.assert_allclose(T2[:3, :3], T[:3, :3], atol=1e-12)
    assert abs(np.linalg.det(T[:3, :3]) - 1) < 1e-12


def test_hold_last_valid_and_stale_reanchor():
    c = HandClutch(neutral(), k=1.0, stale_timeout=1.0)
    c.request_calibration()
    c.update(0, np.zeros(3), np.eye(3), True)
    T1, _ = c.update(0.1, np.array([0.1, 0, 0]), np.eye(3), True)
    T2, i2 = c.update(0.5, valid=False)                 # lost, within timeout
    np.testing.assert_allclose(T2, T1)
    assert not i2["stale"]
    T3, i3 = c.update(1.5, valid=False)
    assert i3["stale"]
    # stale: rate-limited drift back towards the neutral (not frozen, not a jump)
    d1 = np.linalg.norm(T1[:3, 3] - neutral()[:3, 3])
    d3 = np.linalg.norm(T3[:3, 3] - neutral()[:3, 3])
    assert 0 < d1 - d3 <= c.return_lin * 0.2 + 1e-9
    for tt in np.arange(1.6, 12.0, 0.1):
        T3b, _ = c.update(tt, valid=False)
    np.testing.assert_allclose(T3b[:3, 3], neutral()[:3, 3], atol=1e-9)
    # reappears elsewhere after staleness -> re-anchored to the current output, no jump
    T4, i4 = c.update(12.5, np.array([0.5, 0.3, 0.2]), rz(1.0), True)
    np.testing.assert_allclose(T4, T3b, atol=1e-12)
    assert not i4["stale"]
    T5, _ = c.update(12.6, np.array([0.6, 0.3, 0.2]), rz(1.0), True)
    np.testing.assert_allclose(T5[:3, 3] - T3b[:3, 3], [0.1, 0, 0], atol=1e-12)


def test_still_detector():
    from xr_camera_teleop.teleop.clutch import StillDetector
    d = StillDetector(("a",), window=1.0, radius=0.03)
    for i in range(20):
        d.push("a", i * 0.05, [0, 0, 0.005 * (i % 2)])
        assert d.ready() == (i * 0.05 >= 1.0)
    d.push("a", 1.0, [0.2, 0, 0])
    assert not d.ready()
    d.lost("a")
    assert not d.ready()


def test_short_loss_no_reanchor():
    c = HandClutch(neutral(), k=1.0, stale_timeout=1.0)
    c.request_calibration()
    c.update(0, np.zeros(3), np.eye(3), True)
    c.update(0.1, valid=False)
    T, _ = c.update(0.4, np.array([0.2, 0, 0]), np.eye(3), True)
    np.testing.assert_allclose(T[:3, 3] - neutral()[:3, 3], [0.2, 0, 0], atol=1e-12)


def test_disengage_reengage_no_jump():
    c = HandClutch(neutral(), k=1.0)
    c.request_calibration()
    c.update(0, np.zeros(3), np.eye(3), True)
    T1, _ = c.update(0.1, np.array([0.1, 0, 0]), np.eye(3), True)
    c.set_engaged(False)
    T2, _ = c.update(0.2, np.array([0.5, 0.5, 0.5]), rz(1), True)
    np.testing.assert_allclose(T2, T1)
    c.set_engaged(True)
    T3, _ = c.update(0.3, np.array([0.6, 0.5, 0.5]), rz(1), True)
    np.testing.assert_allclose(T3, T1, atol=1e-12)
    T4, _ = c.update(0.4, np.array([0.7, 0.5, 0.5]), rz(1), True)
    np.testing.assert_allclose(T4[:3, 3] - T1[:3, 3], [0.1, 0, 0], atol=1e-12)


def test_max_offset_clamps():
    c = HandClutch(neutral(), k=1.0, max_offset=0.3)
    c.request_calibration()
    c.update(0, np.zeros(3), np.eye(3), True)
    T, _ = c.update(0.1, np.array([5.0, 0, 0]), np.eye(3), True)
    assert np.isclose(np.linalg.norm(T[:3, 3] - neutral()[:3, 3]), 0.3)


def test_tv_const_neutral_values():
    L, R = neutral_from_tv_wrapper_const()
    np.testing.assert_allclose(L[:3, 3], [0.25, 0.15, 0.08], atol=1e-9)
    np.testing.assert_allclose(R[:3, 3], [0.25, -0.15, 0.08], atol=1e-9)
    np.testing.assert_allclose(L[:3, :3], rx(np.pi / 2), atol=1e-9)
    np.testing.assert_allclose(R[:3, :3], rx(-np.pi / 2), atol=1e-9)
