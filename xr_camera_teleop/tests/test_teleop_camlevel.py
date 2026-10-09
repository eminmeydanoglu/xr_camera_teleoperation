import numpy as np

from xr_camera_teleop.teleop.camlevel import level_from_down, level_rotation, pitch_rotation


def test_pitch_matches_gravity_construction():
    th = np.radians(40)
    d_cam = np.array([0, np.cos(th), np.sin(th)])        # camera looking down by th
    np.testing.assert_allclose(level_from_down(d_cam), pitch_rotation(40), atol=1e-12)
    R = pitch_rotation(40)
    np.testing.assert_allclose(R @ np.array([0, 0, 1.0]), [0, np.sin(th), np.cos(th)], atol=1e-12)  # axis points down
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1)


def test_message_overrides_cli_pitch():
    R = pitch_rotation(10)
    np.testing.assert_allclose(level_rotation({"cam_R_level": R.tolist()}, 50), R)
    np.testing.assert_allclose(level_rotation({}, 50), pitch_rotation(50))
    np.testing.assert_allclose(level_rotation(None), np.eye(3))
