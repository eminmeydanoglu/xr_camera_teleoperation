"""Minimal synthetic WiLoR-like hand stream (camera-frame messages per CONTRACT.md).

Used by tests and as a stand-in for xr_camera_teleop/api/replay.py --synthetic.
Pure numpy (+ pyzmq/msgpack only for the optional publisher in __main__).

Hand model: canonical adult hand in the OpenXR wrist frame
(x wrist->middle, y index->pinky for left / pinky->index for right, z back of
hand). `grip` in [0,1] curls the four fingers and tucks the thumb; `pinch`
in [0,1] brings thumb tip to index tip.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

R_W_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# left-hand layout, wrist frame (m). Fingers: MCP offset (x,y), segment lengths
_MCP = {"index": (0.093, -0.032), "middle": (0.095, -0.010), "ring": (0.090, 0.012), "pinky": (0.083, 0.032)}
_SEG = {"index": (0.045, 0.026, 0.021), "middle": (0.050, 0.030, 0.023),
        "ring": (0.046, 0.028, 0.022), "pinky": (0.036, 0.020, 0.019)}
_THUMB_OPEN = np.array([[0.030, -0.018, -0.004], [0.062, -0.040, -0.008],
                        [0.086, -0.058, -0.010], [0.105, -0.072, -0.010]])
_THUMB_FIST = np.array([[0.030, -0.018, -0.004], [0.050, -0.040, -0.030],
                        [0.060, -0.032, -0.052], [0.070, -0.020, -0.056]])


def _ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def canonical_hand_mano21(side="left", grip=0.0, pinch=0.0):
    """(21,3) MANO order keypoints in the OpenXR wrist frame."""
    kp = np.zeros((21, 3))
    fingers = ["index", "middle", "ring", "pinky"]
    tips = {}
    for fi, f in enumerate(fingers):
        base = np.array([_MCP[f][0], _MCP[f][1], 0.0])
        p = base.copy()
        kp[5 + 4 * fi] = p
        ang = grip * np.array([1.35, 1.7, 1.1])
        cum = 0.0
        for j in range(3):
            cum += ang[j]
            p = p + _ry(cum) @ np.array([_SEG[f][j], 0, 0])
            kp[5 + 4 * fi + 1 + j] = p
        tips[f] = p
    th = (1 - grip) * _THUMB_OPEN + grip * _THUMB_FIST
    if pinch > 0:
        target = tips["index"] + np.array([0.0, -0.012, 0.0])
        th = th.copy()
        th[3] = (1 - pinch) * th[3] + pinch * target
        th[2] = (1 - pinch * 0.7) * th[2] + pinch * 0.7 * (0.5 * (th[1] + target))
    kp[1:5] = th
    if side == "right":
        kp[:, 1] *= -1
    return kp


def hand_message_entry(side, p_W, R_W, grip=0.0, pinch=0.0, mano_offset=None, conf=0.9):
    """Entry for msg['hands'][side] given wrist pose in W (R_W = wrist frame)."""
    local = canonical_hand_mano21(side, grip, pinch)
    kpW = local @ R_W.T + p_W
    R_cam_W = R_W_CAM.T
    kp_cam = kpW @ R_cam_W.T
    C = np.eye(3) if mano_offset is None else mano_offset
    return {
        "conf": conf,
        "wrist_pos": kp_cam[0].tolist(),
        "wrist_rot": (R_cam_W @ R_W @ C).tolist(),
        "keypoints": kp_cam.tolist(),
        "keypoints_2d": np.zeros((21, 2)).tolist(),
    }


def make_message(frame_id, t, hands: dict):
    return {"v": 1, "frame_id": frame_id, "t_capture": t, "t_publish": t,
            "image_size": [1280, 720],
            "intrinsics": {"fx": 700.0, "fy": 700.0, "cx": 640.0, "cy": 360.0},
            "source": "replay", "hands": hands}


# comfortable resting spots in W, camera at origin looking +x (operator hands ~0.4 m ahead)
REST = {"left": np.array([0.45, 0.20, -0.30]), "right": np.array([0.45, -0.20, -0.30])}


def rest_rotation(side):
    """Palms facing each other, fingers forward, thumbs up (matches robot neutral)."""
    # wrist frame x fwd; z (back of hand) points outward: left -> +y, right -> -y
    if side == "left":
        return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # Rx(-90)
    return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)      # Rx(+90)


def scripted_pose(t, side):
    """Scripted demo: forward/back, up/down, side sweep and grip cycles."""
    sgn = 1.0 if side == "left" else -1.0
    ph = 0.0 if side == "left" else 0.7
    p = REST[side] + np.array([0.10 * np.sin(0.8 * t + ph) + 0.05,
                               sgn * (0.06 * np.sin(0.5 * t + ph) - 0.0),
                               0.10 * np.sin(0.65 * t + ph)])
    ang = 0.35 * np.sin(0.6 * t + ph)
    c, s = np.cos(ang), np.sin(ang)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rz @ rest_rotation(side)
    grip = 0.5 + 0.5 * np.sin(0.9 * t + ph)
    return p, R, float(np.clip(grip * 1.2 - 0.1, 0, 1)), 0.0


def scripted_message(frame_id, t, sides=("left", "right")):
    hands = {}
    for s in sides:
        p, R, g, pi = scripted_pose(t, s)
        hands[s] = hand_message_entry(s, p, R, g, pi)
    return make_message(frame_id, t, hands)


if __name__ == "__main__":  # minimal publisher (tcp://127.0.0.1:5590)
    import msgpack
    import zmq

    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="tcp://127.0.0.1:5590")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args()
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(a.addr)
    t0 = time.monotonic()
    fid = 0
    while True:
        t = time.monotonic()
        pub.send(msgpack.packb(scripted_message(fid, t - t0) | {"t_capture": t, "t_publish": t}))
        fid += 1
        time.sleep(1.0 / a.fps)
