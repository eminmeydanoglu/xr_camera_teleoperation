"""Camera levelling: rotation R_level with  v_level = R_level @ v_cam  (both OpenCV camera axes:
+x right, +y down, +z forward), where the level frame has +y exactly along gravity and the
optical axis' yaw untouched. Consumers apply it BEFORE R_W_CAM (CONTRACT.md).

Sources: per-message "cam_R_level" (ZED IMU, api/sources.py) else a fixed pitch from the CLI
(positive = camera looking DOWN)."""
from __future__ import annotations

import numpy as np


def pitch_rotation(pitch_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    return np.array([[1.0, 0, 0], [0, c, s], [0, -s, c]])


def level_from_down(d_cam) -> np.ndarray:
    """R_level from the gravity direction (unit 'down') expressed in the camera frame."""
    d = np.asarray(d_cam, float)
    d = d / np.linalg.norm(d)
    f = np.array([0.0, 0.0, 1.0]) - d[2] * d
    if np.linalg.norm(f) < 0.2:      # looking (almost) straight down: use image-up as forward
        f = np.array([0.0, -1.0, 0.0]) + d[1] * d
    f /= np.linalg.norm(f)
    return np.stack([np.cross(d, f), d, f])


def level_rotation(msg, pitch_deg: float = 0.0) -> np.ndarray:
    """cam_R_level from the message when present and valid, else the fixed pitch."""
    r = msg.get("cam_R_level") if msg else None
    if r is not None:
        R = np.asarray(r, float)
        if R.shape == (3, 3) and np.all(np.isfinite(R)):
            return R
    return pitch_rotation(pitch_deg)


def yaw_in_cam(yaw_rad: float) -> np.ndarray:
    """Rotation about the level frame's gravity axis (+y down) by yaw, expressed so that
    R_W_CAM @ yaw_in_cam(a) == Rz(a) @ R_W_CAM (W: z up). Compose as yaw_in_cam(a) @ R_level."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])
