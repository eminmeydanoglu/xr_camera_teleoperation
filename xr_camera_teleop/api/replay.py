#!/usr/bin/env python
"""GPU-free publisher of contract-conformant hand messages (runs in data/xr-env).

  replay.py --synthetic [--hz 30] [--duration S]     moving two-hand skeleton
  replay.py --file rec.jsonl [--loop] [--realtime]   replay of server.py --record output

Hand-local frames (documented so consumers can check kp - wrist == R @ local):
  right: WiLoR/MANO canonical frame as observed from the real model:
         fingers along -x, palm outward normal -y, thumb side +z.
  left : local_left = diag(-1,1,1) @ local_right, and R_left = M R M (the WiLoR
         mirror convention used by api/engine.py), so left kp - wrist = R @ local_left.
"""
import argparse
import json
import math
import time

import msgpack
import numpy as np
import zmq

MIRROR = np.diag([-1.0, 1.0, 1.0])
INTRINSICS = {"fx": 955.765, "fy": 956.185, "cx": 644.77, "cy": 359.3145}
IMAGE_SIZE = [1280, 720]

# Right-hand rest skeleton, metres. Fingers: (MCP position, 3 phalanx lengths).
FINGERS = [  # index, middle, ring, pinky
    (np.array([0.090, 0.0, 0.032]), (0.040, 0.022, 0.020)),
    (np.array([0.095, 0.0, 0.011]), (0.044, 0.026, 0.021)),
    (np.array([0.088, 0.0, -0.010]), (0.041, 0.024, 0.020)),
    (np.array([0.078, 0.0, -0.030]), (0.032, 0.018, 0.017)),
]
THUMB_CMC = np.array([0.025, 0.0, 0.022])
THUMB_LEN = (0.040, 0.032, 0.027)            # CMC->MCP, MCP->IP, IP->tip
THUMB_DIR = np.array([0.55, 0.0, 0.83])
THUMB_DIR /= np.linalg.norm(THUMB_DIR)
FLEX_MAX = np.radians([85.0, 100.0, 65.0])   # MCP, PIP, DIP at full closure


def rot(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * k + (1 - math.cos(ang)) * (k @ k)


def hand_local(closure):
    """21x3 right-hand keypoints in the hand-local frame; closure in [0,1]."""
    c = float(np.clip(closure, 0, 1))
    pts = np.zeros((21, 3))
    # thumb 1..4: bends toward the palm (+y) about an axis perpendicular to its direction
    t_axis = np.cross(THUMB_DIR, [0, 1, 0])
    p, d = THUMB_CMC.copy(), THUMB_DIR.copy()
    pts[1] = p
    for k, length in enumerate(THUMB_LEN):
        d = rot(t_axis, c * np.radians(35.0 if k else 25.0)) @ d
        p = p + d * length
        pts[2 + k] = p
    for f, (mcp, lens) in enumerate(FINGERS):
        base = 5 + 4 * f
        pts[base] = mcp
        p, ang = mcp.copy(), 0.0
        for j, length in enumerate(lens):
            ang += c * FLEX_MAX[j] * (1.0 - 0.08 * f)
            p = p + rot([0, 0, 1], ang) @ np.array([length, 0, 0])
            pts[base + 1 + j] = p
    return pts * np.array([-1.0, -1.0, 1.0])   # build with fingers +x/palm +y, store in MANO frame


# Palm facing away from the camera, fingers up, thumb to image-left (right hand):
# columns = images of the (fingers-+x, palm-+y, thumb-z) frame, then D converts to the MANO frame.
R0 = np.stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).T.astype(float) @ np.diag([-1.0, -1.0, 1.0])


def project(points, k=INTRINSICS):
    z = np.maximum(points[:, 2], 1e-6)
    return np.stack([points[:, 0] / z * k["fx"] + k["cx"], points[:, 1] / z * k["fy"] + k["cy"]], 1)


def synthetic_hands(t):
    """Both hands at time t (seconds): {'left': hand, 'right': hand}."""
    hands = {}
    for side, ph in (("right", 0.0), ("left", 1.7)):
        closure = 0.5 + 0.5 * math.sin(0.9 * t + ph)
        local = hand_local(closure)
        # right-hand-equivalent scene; left is its mirror image about x=0
        p = np.array([0.14 + 0.07 * math.sin(0.5 * t + ph),
                      0.04 * math.sin(0.7 * t + 2 * ph) + 0.03,
                      0.50 + 0.07 * math.sin(0.4 * t + ph)])
        R = (rot([0, 1, 0], 0.5 * math.sin(0.35 * t + ph)) @ rot([1, 0, 0], 0.35 * math.sin(0.45 * t + 1 + ph))
             @ rot([0, 0, 1], 0.4 * math.sin(0.3 * t + 2 + ph)) @ R0)
        kp = (R @ local.T).T + p
        if side == "left":
            kp, R = kp @ MIRROR, MIRROR @ R @ MIRROR
        hands[side] = {"conf": 0.9 - 0.04 * math.sin(t + ph), "wrist_pos": kp[0].tolist(),
                       "wrist_rot": R.tolist(), "keypoints": kp.tolist(),
                       "keypoints_2d": project(kp).tolist()}
    return hands


def synthetic_message(frame_id, t, now=None):
    now = time.monotonic() if now is None else now
    return {"v": 1, "frame_id": frame_id, "t_capture": now, "t_publish": now,
            "image_size": list(IMAGE_SIZE), "intrinsics": dict(INTRINSICS), "source": "replay",
            "hands": synthetic_hands(t)}


def run(args):
    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(args.addr)
    time.sleep(0.3)  # let SUBs connect
    period = 1.0 / args.hz
    start = nxt = time.monotonic()
    print(f"[replay] publishing on {args.addr}", flush=True)
    try:
        if args.synthetic:
            fid = 0
            while args.duration is None or time.monotonic() - start < args.duration:
                now = time.monotonic()
                fid += 1
                sock.send(msgpack.packb(synthetic_message(fid, now - start, now), use_bin_type=True))
                nxt += period
                time.sleep(max(0.0, nxt - time.monotonic()))
        else:
            with open(args.file) as f:
                msgs = [json.loads(line) for line in f if line.strip()]
            if not msgs:
                raise SystemExit("empty recording")
            while True:
                t0 = msgs[0]["t_publish"]
                begin = time.monotonic()
                for m in msgs:
                    if args.realtime:
                        time.sleep(max(0.0, begin + (m["t_publish"] - t0) - time.monotonic()))
                    else:
                        time.sleep(max(0.0, nxt - time.monotonic()))
                        nxt += period
                    now = time.monotonic()
                    m = {**m, "t_capture": now, "t_publish": now, "source": "replay"}
                    sock.send(msgpack.packb(m, use_bin_type=True))
                if not args.loop or (args.duration and time.monotonic() - start > args.duration):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--file", help="jsonl produced by server.py --record")
    ap.add_argument("--addr", default="tcp://127.0.0.1:5590")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="--file: use recorded timing instead of --hz")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
