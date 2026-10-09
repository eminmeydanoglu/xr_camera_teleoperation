import sys
from pathlib import Path

import msgpack
import numpy as np
import pytest

API = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API))
import engine  # noqa: E402
import replay  # noqa: E402

M = np.diag([-1.0, 1.0, 1.0])


def validate(msg):
    assert msg["v"] == 1
    assert isinstance(msg["frame_id"], int)
    assert msg["t_publish"] >= msg["t_capture"] - 1e-9
    assert len(msg["image_size"]) == 2
    assert set(msg["intrinsics"]) == {"fx", "fy", "cx", "cy"}
    assert msg["source"] in ("live", "replay")
    assert set(msg["hands"]) <= {"left", "right"}
    for h in msg["hands"].values():
        assert 0.0 <= h["conf"] <= 1.0
        assert np.shape(h["wrist_pos"]) == (3,)
        R = np.asarray(h["wrist_rot"])
        assert R.shape == (3, 3)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.linalg.det(R) == pytest.approx(1, abs=1e-6)
        kp = np.asarray(h["keypoints"])
        assert kp.shape == (21, 3) and np.isfinite(kp).all()
        assert np.allclose(kp[0], h["wrist_pos"])
        assert np.asarray(h["keypoints_2d"]).shape == (21, 2)


def test_synthetic_schema_and_roundtrip():
    for i, t in enumerate(np.linspace(0, 30, 61)):
        msg = replay.synthetic_message(i, t)
        validate(msg)
        validate(msgpack.unpackb(msgpack.packb(msg, use_bin_type=True), raw=False))
        assert set(msg["hands"]) == {"left", "right"}


def test_synthetic_plausible_ranges():
    zs, closes = [], []
    for t in np.linspace(0, 40, 200):
        for h in replay.synthetic_hands(t).values():
            zs.append(h["wrist_pos"][2])
            kp = np.asarray(h["keypoints"])
            closes.append(np.linalg.norm(kp[8] - kp[0]))
            assert np.linalg.norm(kp[9] - kp[0]) == pytest.approx(0.0957, abs=0.01)  # palm length const
    assert 0.38 <= min(zs) and max(zs) <= 0.62
    assert max(closes) - min(closes) > 0.04   # hand really opens/closes


def test_left_right_sides_and_mirror_convention():
    h = replay.synthetic_hands(3.0)
    assert h["right"]["wrist_pos"][0] > 0 > h["left"]["wrist_pos"][0]
    # kp - wrist = R @ local  (left uses mirrored local frame)
    for side in ("left", "right"):
        kp, R = np.asarray(h[side]["keypoints"]), np.asarray(h[side]["wrist_rot"])
        loc = (R.T @ (kp - kp[0]).T).T
        if side == "left":
            loc = loc @ M
        # recovered local frame must be rigid right-hand skeleton: bone lengths match
        ref = replay.hand_local(0.5 + 0.5 * np.sin(0.9 * 3.0 + (1.7 if side == "left" else 0.0)))
        assert np.allclose(loc, ref - ref[0], atol=1e-9)


def test_resolve_wrist_rot_mirror_is_proper():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        for right in (True, False):
            r = engine.resolve_wrist_rot(q, right)
            assert np.linalg.det(r) == pytest.approx(1.0) and np.allclose(r @ r.T, np.eye(3))
        assert np.allclose(engine.resolve_wrist_rot(q, True), q)
        assert np.allclose(engine.resolve_wrist_rot(q, False), M @ q @ M)


def test_mirror_consistency_with_flipped_keypoints():
    # network space (flipped crop): j_f = R local. Real left: M j_f = (M R M)(M local)
    loc = replay.hand_local(0.3)
    q = replay.rot([0.3, 1.0, 0.2], 0.9)
    j_f = (q @ loc.T).T
    real = j_f * np.array([-1, 1, 1])
    R = engine.resolve_wrist_rot(q, False)
    assert np.allclose(real, (R @ (loc @ M).T).T)


def test_nearest_so3_fixes_noise():
    rng = np.random.default_rng(1)
    r = engine.nearest_so3(replay.rot([1, 2, 3], 0.7) + 1e-3 * rng.normal(size=(3, 3)))
    assert np.linalg.det(r) == pytest.approx(1) and np.allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(engine.nearest_so3(np.diag([1.0, 1, -1]))) == pytest.approx(1)


def test_hand_message_projection():
    k = {"fx": 900.0, "fy": 900.0, "cx": 640.0, "cy": 360.0}
    kp = replay.hand_local(0.0) + np.array([0.1, -0.05, 0.5])
    m = engine.hand_message("right", 0.8, kp, np.eye(3), k)
    assert np.allclose(m["keypoints_2d"][0], [640 + 900 * 0.1 / 0.5, 360 - 900 * 0.05 / 0.5])
