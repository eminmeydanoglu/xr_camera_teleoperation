import numpy as np

from xr_camera_teleop.teleop import filters as F


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def test_quat_roundtrip_and_slerp():
    rng = np.random.default_rng(0)
    for _ in range(50):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        np.testing.assert_allclose(F.quat_to_mat(F.mat_to_quat(Q)), Q, atol=1e-9)
        v = F.rot_log(Q)
        np.testing.assert_allclose(F.rot_exp(v), Q, atol=1e-8)
    q = F.slerp = F.quat_slerp(F.mat_to_quat(np.eye(3)), F.mat_to_quat(rz(1.0)), 0.5)
    np.testing.assert_allclose(F.quat_to_mat(q), rz(0.5), atol=1e-9)


def test_oneeuro_reduces_jitter_but_tracks_motion():
    rng = np.random.default_rng(0)
    f = F.OneEuroFilter(1.0, 0.5)
    ts = np.arange(0, 3, 1 / 30)
    sig = np.stack([np.sin(0.5 * ts) * 0.1, 0 * ts, 0 * ts], 1)
    noisy = sig + rng.normal(scale=0.01, size=sig.shape)
    out = np.array([f(x, t) for x, t in zip(noisy, ts)])
    assert np.std(out - sig) < np.std(noisy - sig)
    # step: converges
    f.reset()
    for t in np.arange(0, 2, 1 / 30):
        y = f(np.array([1.0, 0, 0]), t)
    assert abs(y[0] - 1.0) < 1e-3
    # fast motion lag is small (beta effect)
    f = F.OneEuroFilter(1.0, 5.0)
    for t in np.arange(0, 1, 1 / 30):
        y = f(np.array([2.0 * t, 0, 0]), t)
    assert abs(y[0] - 2.0 * 29 / 30) < 0.12


def test_so3_filter_sign_continuity_and_smoothing():
    f = F.SO3OneEuroFilter(1.0, 0.3)
    ts = np.arange(0, 2, 1 / 30)
    prev = None
    for i, t in enumerate(ts):
        R = rz(0.6 * np.sin(2 * t))
        q = F.mat_to_quat(R)
        if i % 2:                      # inject double-cover flips on the raw side
            q = -q
        out = f(F.quat_to_mat(q), t)
        assert np.allclose(out @ out.T, np.eye(3), atol=1e-9)
        if prev is not None:
            assert F.rot_log(out @ prev.T).__abs__().max() < 0.2   # no jump
        prev = out
    # converges to constant target
    f.reset()
    for t in np.arange(0, 3, 1 / 30):
        out = f(rz(1.2), t)
    assert np.linalg.norm(F.rot_log(out @ rz(1.2).T)) < 1e-3


def test_so3_filter_near_pi_hemisphere():
    f = F.SO3OneEuroFilter()
    a = rz(np.pi - 0.01)
    b = rz(-np.pi + 0.01)           # 0.02 rad apart through the cut
    f(a, 0.0)
    out = f(b, 1 / 30)
    assert np.linalg.norm(F.rot_log(out @ a.T)) < 0.05


def test_velocity_limits():
    p, R = np.zeros(3), np.eye(3)
    p2, R2 = F.limit_pose_step(p, R, np.array([1.0, 0, 0]), rz(2.0), 0.01, 1.0, 3.0)
    assert np.isclose(np.linalg.norm(p2), 0.01)
    assert np.isclose(np.linalg.norm(F.rot_log(R2)), 0.03, atol=1e-6)
    p3, R3 = F.limit_pose_step(p, R, np.array([0.001, 0, 0]), rz(0.001), 0.01, 1.0, 3.0)
    np.testing.assert_allclose(p3, [0.001, 0, 0])
    lim = F.PoseRateLimiter(1.0, 3.0)
    lim(np.zeros(3), np.eye(3), 0.0)
    pp, _ = lim(np.array([5.0, 0, 0]), np.eye(3), 0.1)
    assert np.isclose(pp[0], 0.1)
