"""Synthetic contract-message generator (own copy; independent of agent A's replay.py)."""
from __future__ import annotations

import argparse
import time

import numpy as np

SEG = [0.04, 0.028, 0.02]
# finger base offsets in hand-local frame (right hand; x mirrored for left)
BASES = [(-0.035, 0.015), (-0.02, 0.09), (0.0, 0.095), (0.02, 0.09), (0.04, 0.08)]
R_PALM_BACK_TO_CAM = np.diag([1.0, -1.0, -1.0])  # local +y -> cam up, fingers curl to cam +z


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def local_hand(side, curl):
    """21x3 keypoints in hand-local frame. curl in [0,1]. Fingers along +y, curl toward -z."""
    kp = np.zeros((21, 3))
    for f in range(5):
        bx, by = BASES[f]
        x = bx * (1.0 if side == "right" else -1.0)
        base = np.array([x, by, 0.0])
        if f == 0:
            d0 = np.array([-0.5 if side == "right" else 0.5, 0.85, 0.0])
            d0 /= np.linalg.norm(d0)
            segs = [0.04, 0.032, 0.026]
            bend = [0.2, 0.9, 0.9]
        else:
            d0 = np.array([0.0, 1.0, 0.0])
            segs = SEG
            bend = [1.6, 1.7, 1.4]
        p, ang = base.copy(), 0.0
        kp[1 + 4 * f] = p
        for j in range(3):
            ang += bend[j] * curl
            d = _rotx(-ang) @ d0  # rotating about x by -ang tilts +y toward -z
            p = p + segs[j] * d
            kp[2 + 4 * f + j] = p
    return kp


def make_message(frame_id, t, image_size=(1280, 720)):
    curl = 0.5 + 0.5 * np.sin(2 * np.pi * t / 4.0)
    hands = {}
    for side, x, ph in (("left", -0.18, 0.0), ("right", 0.18, 1.0)):
        c = curl if side == "left" else 0.5 + 0.5 * np.sin(2 * np.pi * t / 4.0 + ph)
        R = R_PALM_BACK_TO_CAM @ _rotx(0.0)
        wrist = np.array([x + 0.03 * np.sin(t), 0.02 * np.cos(t), 0.55])
        kp = local_hand(side, c) @ R.T + wrist
        hands[side] = {
            "conf": 0.95,
            "wrist_pos": wrist.tolist(),
            "wrist_rot": R.tolist(),
            "keypoints": kp.tolist(),
            "keypoints_2d": (kp[:, :2] * 800 / kp[:, 2:3] + [640, 360]).tolist(),
        }
    now = time.monotonic()
    return {"v": 1, "frame_id": frame_id, "t_capture": now, "t_publish": now,
            "image_size": list(image_size),
            "intrinsics": {"fx": 800.0, "fy": 800.0, "cx": 640.0, "cy": 360.0},
            "source": "replay", "hands": hands}


def main():
    import msgpack
    import zmq

    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="tcp://127.0.0.1:5590")
    ap.add_argument("--fps", type=float, default=30)
    a = ap.parse_args()
    s = zmq.Context.instance().socket(zmq.PUB)
    s.bind(a.addr)
    t0, i = time.monotonic(), 0
    while True:
        s.send(msgpack.packb(make_message(i, time.monotonic() - t0)))
        i += 1
        time.sleep(1 / a.fps)


if __name__ == "__main__":
    main()
