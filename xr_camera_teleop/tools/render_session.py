#!/usr/bin/env python
"""Offline render of a recorded session into ONE side-by-side mp4 (run with data/xr-env).

    data/xr-env/bin/python xr_camera_teleop/tools/render_session.py \
        --session data/sessions/20261009_105318 [--out X.mp4] [--k 0.8] [--start S --duration S]

<session> is the base timestamp (or a path ending in it; cam_/hands_ prefixes are found in the directory).
Panels (frame-synchronous, 30 fps, real time):
  LEFT   camera frame + WiLoR overlay (api/overlay.draw_overlay)
  MIDDLE MuJoCo floating hands, relative mode, robot-neutral anchors (viz/), calibrated at the first
         frame with both hands
  RIGHT  G1 MuJoCo: WiLoRTeleWrapper (clutch, filters; fed the recorded msgs with their recorded
         timestamps through a fake client/clock) -> G1_29_ArmIK -> Dex3 HandRetargeting, per-joint
         velocity limit (teleop/g1_mujoco_teleop.G1Sim), standing, pelvis fixed.
Frames without a message reuse the previous message (the wrapper sees the same frame_id: nothing new).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PW, PH, BAR = 640, 360, 28
TITLES = ("WiLoR", "Floating hands", "G1 (xr_teleoperate IK + Dex3)")
HIDE_AFTER, STALE_AFTER = 2.0, 0.3


def resolve(session):
    p = Path(session)
    base = p.name
    for pre in ("cam_", "hands_", "render_"):
        if base.startswith(pre):
            base = base[len(pre):]
    base = Path(base).stem if base.endswith((".mkv", ".csv", ".jsonl", ".json", ".mp4")) else base
    d = p.parent
    return d, base


class ReplayClient:
    """Fake HandStreamClient: returns the message pushed for the current video frame."""

    def __init__(self):
        self.msg = None

    def latest(self, max_age=0.3):
        return self.msg

    def close(self):
        pass


def put_text(img, s, org, scale=0.55, color=(255, 255, 255), th=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, th, cv2.LINE_AA)


def make_cam(lookat, dist, az, el):
    import mujoco
    c = mujoco.MjvCamera()
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--k", type=float, default=0.8)
    ap.add_argument("--start", type=float, default=0.0, help="seconds from session start")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0,
                    help="camera pitch, positive = looking down (messages carry no cam_R_level in old recordings)")
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args(argv)

    d, base = resolve(a.session)
    mkv, csvp = d / f"cam_{base}.mkv", d / f"cam_{base}.csv"
    jl = d / f"hands_{base}.jsonl"
    out = Path(a.out) if a.out else d / f"render_{base}.mp4"

    rows = list(csv.DictReader(open(csvp)))
    fids = [int(r["frame_id"]) for r in rows]
    ts = np.array([float(r["t_capture"]) for r in rows])
    msgs = {}
    for line in open(jl):
        m = json.loads(line)
        msgs[m["frame_id"]] = m
    n_total = len(rows)
    t0 = ts[0]

    import mujoco
    from xr_camera_teleop.api.overlay import draw_overlay
    from xr_camera_teleop.teleop import xrt_compat
    from xr_camera_teleop.teleop.clutch import neutral_from_ik_fk
    from xr_camera_teleop.teleop.g1_mujoco_teleop import G1Sim, solve_hand
    from xr_camera_teleop.teleop.wilor_tv_wrapper import SIDES, WiLoRTeleWrapper
    from xr_camera_teleop.viz.hands_scene import HandsScene, default_camera
    from xr_camera_teleop.viz.mapping import WAIST_DROP, HandMapper, robot_anchors

    # ---- G1 pipeline (same construction as g1_mujoco_teleop.main)
    ik = xrt_compat.make_arm_ik()
    hr = xrt_compat.make_hand_retargeting()
    arm_names = list(ik.reduced_robot.model.names[1:])
    hand_names = {"left": list(hr.left_retargeting_joint_names), "right": list(hr.right_retargeting_joint_names)}
    sim = G1Sim(arm_names, hand_names)
    clk = [0.0]
    client = ReplayClient()
    wrapper = WiLoRTeleWrapper(client=client, k=a.k, neutral=neutral_from_ik_fk(ik), clock=lambda: clk[0],
                               auto_calibrate=True, cam_pitch_deg=a.cam_pitch_deg)
    xrt_compat.connect_ik(ik, wrapper, lambda: clk[0])
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, PW)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, PH)
    r_g1 = mujoco.Renderer(sim.model, PH, PW)
    cam_g1 = make_cam([0.15, 0.0, 1.0], 1.15, 150, -8)

    # ---- floating hands (relative, --neutral robot semantics)
    cam_h = 1.5
    waist_h = cam_h - WAIST_DROP
    scene = HandsScene(cam_h, waist_height=waist_h)
    mapper = HandMapper("relative", a.k, cam_h, neutral_rot="ref", anchors=robot_anchors(waist_h),
                        max_depth=0.9)
    r_h = mujoco.Renderer(scene.model, PH, PW)
    look, dist, az, el = default_camera(cam_h)
    cam_hands = make_cam(look - np.array([0.0, 0.0, 0.15]), dist * 1.5, az, el)
    last_cam, last_seen = {}, {"left": -1e9, "right": -1e9}

    # ---- video in / out
    cap = cv2.VideoCapture(str(mkv))
    W, H = int(cap.get(3)), int(cap.get(4))
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{3*PW}x{PH+BAR}",
         "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(a.crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)

    t_start = a.start
    t_end = t_start + a.duration if a.duration else 1e18
    cur_msg, prev_t = None, None
    stats = {"frames": 0, "both": 0, "rendered": 0, "ik_fail": 0, "ik_ms": [], "pos_err_mm": [],
             "calib_frame": None, "no_msg": 0, "missing_left": 0, "missing_right": 0}
    cur_dq = np.zeros(14)
    calibrated_at = None
    wall0 = time.time()
    for i in range(n_total):
        ok, frame = cap.read()
        if not ok:
            break
        t = ts[i] - t0
        if t > t_end:
            break
        m = msgs.get(fids[i])
        if m is None:
            stats["no_msg"] += 1
            m = cur_msg
        cur_msg = m if m is not None else cur_msg
        has_hands = cur_msg.get("hands", {}) if cur_msg else {}
        render = t >= t_start
        # ---------- pipeline (always, from the beginning, so state matches the full-session render)
        now = float(ts[i])
        clk[0] = now
        client.msg = cur_msg
        dt = float(np.clip(now - prev_t, 1 / 60, 0.1)) if prev_t is not None else 1 / 30
        prev_t = now
        # floating hands
        if m is not None and m is not None and fids[i] == m["frame_id"]:
            for h, dd in m.get("hands", {}).items():
                if dd["wrist_pos"][2] > m.get("max_hand_depth", 0.9):   # not the operator's hand (same gate as the wrapper)
                    continue
                last_cam[h] = (np.asarray(dd["wrist_pos"], float), np.asarray(dd["wrist_rot"], float),
                               np.asarray(dd["keypoints"], float))
                last_seen[h] = now
        # G1
        td = wrapper.get_tele_data()
        mapper.R_level = wrapper.effective_level(cur_msg)   # after the wrapper: incl. heading set at calibration
        if calibrated_at is None and wrapper.auto_cal_t is not None:   # same moment as the G1 clutch
            mapper.calibrate({h: (last_cam[h][0], last_cam[h][1]) for h in SIDES})
            calibrated_at = t
            stats["calib_frame"] = i
            stats["calib_t"] = round(float(t), 2)
        import time as _t
        t1 = _t.perf_counter()
        sol_q, _ = ik.solve_ik(td.left_wrist_pose, td.right_wrist_pose, sim.cur_arm.copy(), cur_dq)
        stats["ik_ms"].append((_t.perf_counter() - t1) * 1e3)
        arm_q = None
        if np.all(np.isfinite(sol_q)):
            arm_q = np.asarray(sol_q, float)
        else:
            stats["ik_fail"] += 1
        hand_q = {}
        if td.motion_data_ready:
            for s, hp in (("left", td.left_hand_pos), ("right", td.right_hand_pos)):
                q = solve_hand(hr, s, hp)
                if np.all(np.isfinite(q)):
                    hand_q[s] = q
        prev = sim.cur_arm.copy()
        sim.step_targets(arm_q, hand_q, dt)
        cur_dq = (sim.cur_arm - prev) / dt
        stats["frames"] += 1
        stats["both"] += int(all(h in has_hands for h in SIDES) and m is not None and fids[i] == m["frame_id"])
        if not render:
            continue
        # ---------- render
        stats["rendered"] += 1
        # left
        intr = (cur_msg or {}).get("intrinsics") or {"fx": 1, "fy": 1, "cx": 0, "cy": 0}
        pm = {"intrinsics": intr, "hands": has_hands}
        lat = (cur_msg["t_publish"] - cur_msg["t_capture"]) * 1e3 if cur_msg else 0.0
        left = draw_overlay(frame, pm, {"fps": 30.0, "infer_ms": 0.0, "latency_ms": lat})
        left = cv2.resize(left, (PW, PH), interpolation=cv2.INTER_AREA)
        # middle
        for h in SIDES:
            age = now - last_seen[h]
            res = mapper.map(h, *last_cam[h]) if (h in last_cam and age <= HIDE_AFTER and mapper.calibrated) else None
            if res is None:
                scene.set_state(h, "hidden")
                continue
            scene.set_state(h, "live" if age < STALE_AFTER else "stale")
            scene.pose(h, *res)
        scene.forward()
        r_h.update_scene(scene.data, cam_hands)
        mid = cv2.cvtColor(r_h.render(), cv2.COLOR_RGB2BGR)
        put_text(mid, "calibrated" if calibrated_at is not None else "waiting: both hands still for 1 s", (10, PH - 12),
                 0.5, (120, 255, 120) if calibrated_at is not None else (120, 200, 255))
        # right
        r_g1.update_scene(sim.data, cam_g1)
        rgt = cv2.cvtColor(r_g1.render(), cv2.COLOR_RGB2BGR)
        put_text(rgt, f"clutch k={a.k}  cam pitch {a.cam_pitch_deg:g} deg", (10, PH - 12), 0.5)
        row = np.concatenate([left, mid, rgt], axis=1)
        bar = np.full((BAR, 3 * PW, 3), 32, np.uint8)
        for j, ttl in enumerate(TITLES):
            put_text(bar, ttl, (j * PW + 10, 20), 0.6)
            if j:
                bar[:, j * PW - 1:j * PW + 1] = 90
        put_text(bar, f"t={t:6.2f}s  f{i}", (3 * PW - 260, 20), 0.5, (200, 255, 200))
        enc.stdin.write(np.ascontiguousarray(np.concatenate([bar, row], axis=0)).tobytes())
        if stats["rendered"] % 100 == 0:
            print(f"[render] {stats['rendered']} frames, t={t:.1f}s, {time.time()-wall0:.0f}s wall", flush=True)
    enc.stdin.close()
    enc.wait()
    cap.release()
    ik_ms = np.array(stats.pop("ik_ms"))
    stats["ik_ms_mean"] = float(ik_ms.mean()) if len(ik_ms) else None
    stats["both_pct"] = 100.0 * stats["both"] / max(stats["frames"], 1)
    stats["wall_s"] = time.time() - wall0
    stats["out"] = str(out)
    print("[summary] " + json.dumps(stats))
    sys.stdout.flush()
    os._exit(0)  # avoid noisy EGL teardown


if __name__ == "__main__":
    main()
