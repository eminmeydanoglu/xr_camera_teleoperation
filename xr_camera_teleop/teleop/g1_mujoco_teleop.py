#!/usr/bin/env python
"""Standalone MuJoCo G1 (Dex3) teleop driven by WiLoR via xr_teleoperate's own IK + retargeting.

    TeleData = WiLoRTeleWrapper.get_tele_data()
      -> G1_29_ArmIK.solve_ik(left_wrist_pose, right_wrist_pose, q, dq)   (14 arm joints)
      -> HandRetargeting (dex-retargeting DexPilot, Dex3 yml)              (7+7 hand joints)
    applied kinematically to assets/g1/g1_body29_hand14.xml (pelvis frozen, standing),
    with per-joint velocity limits. No DDS / unitree_sdk2.

Run (live, needs xr_camera_teleop/api/server.py publishing on tcp://127.0.0.1:5590):
    data/xr-env/bin/python -m xr_camera_teleop.teleop.g1_mujoco_teleop --addr tcp://127.0.0.1:5590
Viewer keys: C = calibrate (clutch: current hands <-> robot neutral), Space = hold/release
(clutch pedal; release re-anchors without a jump), R = reset calibration.
Headless verification with built-in synthetic hands:
    data/xr-env/bin/python -m xr_camera_teleop.teleop.g1_mujoco_teleop --synthetic --headless \
        --frames 300 --snapshot /tmp/g1.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                       # allow `python path/to/file.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "xr_camera_teleop.teleop"

import numpy as np

from . import xrt_compat
from .clutch import neutral_from_ik_fk, neutral_from_tv_wrapper_const
from .filters import rot_log
from .wilor_tv_wrapper import SIDES, WiLoRTeleWrapper

ARM_VMAX = 4.0     # rad/s joint velocity limit on applied arm targets
HAND_VMAX = 8.0


class SyntheticClient:
    """Stand-in for HandStreamClient: scripted two-hand motion in-process."""

    def __init__(self, clock):
        self.clock, self.t0, self.fid = clock, clock(), 0

    def latest(self, max_age=0.3):
        from .synthetic_stream import scripted_message
        t = self.clock()
        self.fid += 1
        msg = scripted_message(self.fid, t - self.t0)
        msg["t_capture"] = msg["t_publish"] = t
        return msg

    def close(self):
        pass


def _lazy_mujoco():
    import mujoco
    return mujoco


class G1Sim:
    def __init__(self, arm_names, retarget_names):
        mujoco = _lazy_mujoco()
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xrt_compat.G1_MJCF))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = self.model.qpos0
        self.qadr, self.lim = {}, {}
        for i in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if n:
                self.qadr[n] = self.model.jnt_qposadr[i]
                self.lim[n] = self.model.jnt_range[i].copy() if self.model.jnt_limited[i] else None
        for n in list(arm_names) + [x for v in retarget_names.values() for x in v]:
            if n not in self.qadr:
                raise KeyError(f"joint {n} not in MJCF")        # by-name mapping, fail loudly
        self.arm_names = list(arm_names)
        self.hand_names = retarget_names
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        mujoco.mj_forward(self.model, self.data)
        self.cur_arm = np.zeros(len(arm_names))
        self.cur_hand = {s: np.zeros(len(retarget_names[s])) for s in retarget_names}

    def _clip(self, names, q):
        out = q.copy()
        for i, n in enumerate(names):
            if self.lim.get(n) is not None:
                out[i] = np.clip(out[i], *self.lim[n])
        return out

    def step_targets(self, arm_q, hand_q, dt):
        """Rate-limited kinematic update; legs/waist stay at qpos0, pelvis frozen."""
        if arm_q is not None:
            tgt = self._clip(self.arm_names, np.asarray(arm_q))
            self.cur_arm += np.clip(tgt - self.cur_arm, -ARM_VMAX * dt, ARM_VMAX * dt)
        for s, q in (hand_q or {}).items():
            if q is not None:
                tgt = self._clip(self.hand_names[s], np.asarray(q))
                self.cur_hand[s] += np.clip(tgt - self.cur_hand[s], -HAND_VMAX * dt, HAND_VMAX * dt)
        for i, n in enumerate(self.arm_names):
            self.data.qpos[self.qadr[n]] = self.cur_arm[i]
        for s in self.hand_names:
            for i, n in enumerate(self.hand_names[s]):
                self.data.qpos[self.qadr[n]] = self.cur_hand[s][i]
        self.data.qvel[:] = 0
        self.mj.mj_forward(self.model, self.data)

    def pelvis_T(self):
        T = np.eye(4)
        T[:3, :3] = self.data.xmat[self.pelvis_id].reshape(3, 3)
        T[:3, 3] = self.data.xpos[self.pelvis_id]
        return T

    def add_markers(self, scn, targets):
        mj = self.mj
        Tp = self.pelvis_T()
        cols = {"left": (0.1, 0.9, 0.2, 0.8), "right": (0.95, 0.5, 0.1, 0.8)}
        for s, T in targets.items():
            if scn.ngeom >= scn.maxgeom:
                return
            Tw = Tp @ T
            g = scn.geoms[scn.ngeom]
            mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_SPHERE, np.array([0.025, 0, 0]), Tw[:3, 3],
                            Tw[:3, :3].flatten(), np.array(cols[s], dtype=np.float32))
            scn.ngeom += 1
            for ax, c in enumerate(([1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1])):
                if scn.ngeom >= scn.maxgeom:
                    return
                g = scn.geoms[scn.ngeom]
                a = Tw[:3, ax]
                mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9),
                                np.array(c, dtype=np.float32))
                mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.004, Tw[:3, 3], Tw[:3, 3] + 0.08 * a)
                scn.ngeom += 1


def render_snapshot(sim, path, targets=None, w=960, h=720, azimuth=150, elevation=-12, dist=1.7):
    mujoco = sim.mj
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, w)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, h)
    try:
        r = mujoco.Renderer(sim.model, h, w)
    except Exception as e:                             # EGL unavailable -> retry with osmesa
        if os.environ.get("MUJOCO_GL") != "osmesa":
            print(f"[render] EGL failed ({e}); re-exec with MUJOCO_GL=osmesa", file=sys.stderr)
            os.environ["MUJOCO_GL"] = "osmesa"
            os.execv(sys.executable, [sys.executable] + sys.argv)
        raise
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.25, 0.0, 1.05]
    cam.distance, cam.azimuth, cam.elevation = dist, azimuth, elevation
    r.update_scene(sim.data, camera=cam)
    if targets:
        sim.add_markers(r.scene, targets)
    img = r.render()
    r.close()
    from PIL import Image
    Image.fromarray(img).save(path)


def solve_hand(hr, side, hand_pos):
    idx = hr.left_indices if side == "left" else hr.right_indices
    rt = hr.left_retargeting if side == "left" else hr.right_retargeting
    ref = hand_pos[idx[1, :]] - hand_pos[idx[0, :]]          # robot_hand_unitree.py:192-193
    return rt.retarget(ref)                                   # order == rt.joint_names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default="tcp://127.0.0.1:5590")
    ap.add_argument("--k", type=float, nargs="+", default=[0.8], help="position gain (1 or 3 values x y z)")
    ap.add_argument("--arm-only", action="store_true")
    ap.add_argument("--hands-only", action="store_true")
    ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--neutral", choices=["fk", "tv_const"], default="fk")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="use in-process scripted hands instead of ZMQ")
    ap.add_argument("--frames", type=int, default=0, help="stop after N ticks (0 = run until closed)")
    ap.add_argument("--snapshot", default=None, help="PNG path (suffix _fNNNN added with --snapshot-every)")
    ap.add_argument("--snapshot-every", type=int, default=0)
    ap.add_argument("--auto-calibrate", action="store_true", help="calibrate when both hands are first seen")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0, help="camera pitch, + = looking down")
    ap.add_argument("--no-filters", action="store_true")
    ap.add_argument("--wrist-source", choices=["keypoints", "mano"], default="keypoints")
    ap.add_argument("--stats-json", default=None)
    a = ap.parse_args(argv)
    if a.headless and a.snapshot is None and a.frames == 0:
        ap.error("--headless needs --frames (and usually --snapshot)")
    virtual = a.headless and a.synthetic
    if a.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    dt = 1.0 / a.hz
    vt = [1000.0]
    clock = (lambda: vt[0]) if virtual else time.monotonic

    t0 = time.time()
    ik = xrt_compat.make_arm_ik()
    hr = xrt_compat.make_hand_retargeting()
    arm_names = list(ik.reduced_robot.model.names[1:])
    hand_names = {"left": list(hr.left_retargeting_joint_names), "right": list(hr.right_retargeting_joint_names)}
    sim = G1Sim(arm_names, hand_names)
    print(f"[init] IK+retargeting+MuJoCo ready in {time.time() - t0:.1f}s; arm joints={arm_names}")

    neutral = neutral_from_ik_fk(ik) if a.neutral == "fk" else neutral_from_tv_wrapper_const()
    client = SyntheticClient(clock) if a.synthetic else None
    k = a.k[0] if len(a.k) == 1 else np.array(a.k)
    wrapper = WiLoRTeleWrapper(addr=a.addr, client=client, k=k, neutral=neutral, clock=clock,
                               use_filters=not a.no_filters, wrist_source=a.wrist_source, cam_pitch_deg=a.cam_pitch_deg,
                               auto_calibrate=a.auto_calibrate or (a.synthetic and a.headless))
    xrt_compat.connect_ik(ik, wrapper, clock)

    ev = {"calibrate": False, "hold": False, "reset": False}
    state = {"held": False}

    def key_cb(keycode):
        ch = chr(keycode) if 0 <= keycode < 256 else ""
        if ch in ("C", "c"):
            ev["calibrate"] = True
        elif keycode == 32:
            ev["hold"] = True
        elif ch in ("R", "r"):
            ev["reset"] = True

    viewer = None
    if not a.headless:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data, key_callback=key_cb)
        viewer.cam.lookat[:] = [0.25, 0.0, 1.05]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 150, -12
        print("[keys] C = calibrate   Space = hold/release   R = reset calibration")

    rec = {"ik_ms": [], "hand_ms": [], "tick_ms": [], "pos_err_mm": [], "rot_err_deg": [],
           "nan": 0, "arm_q_max": 0.0, "hq": {"left": [], "right": []}}
    cur_dq = np.zeros(14)
    left_ready = right_ready = False
    n = 0
    next_t = time.monotonic()
    try:
        while True:
            if viewer is not None and not viewer.is_running():
                break
            if a.frames and n >= a.frames:
                break
            tick0 = time.perf_counter()
            if ev["calibrate"]:
                ev["calibrate"] = False
                wrapper.calibrate()
                print("[clutch] calibrate requested")
            if ev["reset"]:
                ev["reset"] = False
                wrapper.reset()
                print("[clutch] reset")
            if ev["hold"]:
                ev["hold"] = False
                state["held"] = not state["held"]
                wrapper.set_engaged(not state["held"])
                print("[clutch] HOLD" if state["held"] else "[clutch] released")

            td = wrapper.get_tele_data()
            tgt = {"left": td.left_wrist_pose, "right": td.right_wrist_pose}
            arm_q = None
            if not a.hands_only:
                t1 = time.perf_counter()
                sol_q, _ = ik.solve_ik(td.left_wrist_pose, td.right_wrist_pose, sim.cur_arm.copy(), cur_dq)
                rec["ik_ms"].append((time.perf_counter() - t1) * 1e3)
                if not np.all(np.isfinite(sol_q)):
                    rec["nan"] += 1
                else:
                    arm_q = np.asarray(sol_q, float)
                    rec["arm_q_max"] = max(rec["arm_q_max"], float(np.abs(arm_q).max()))
                    pe, re = _fk_err(ik, arm_q, td)
                    rec["pos_err_mm"].append(pe * 1e3)
                    rec["rot_err_deg"].append(np.degrees(re))
            hand_q = {}
            if not a.arm_only and td.motion_data_ready:
                t1 = time.perf_counter()
                for s, hp in (("left", td.left_hand_pos), ("right", td.right_hand_pos)):
                    q = solve_hand(hr, s, hp)
                    if np.all(np.isfinite(q)):
                        hand_q[s] = q
                        rec["hq"][s].append(q)
                    else:
                        rec["nan"] += 1
                rec["hand_ms"].append((time.perf_counter() - t1) * 1e3)
            prev = sim.cur_arm.copy()
            sim.step_targets(arm_q, hand_q, dt)
            cur_dq = (sim.cur_arm - prev) / dt

            if a.snapshot and (a.snapshot_every and n % a.snapshot_every == 0 and n > 0):
                render_snapshot(sim, _snap_name(a.snapshot, n), tgt)
            if viewer is not None:
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    sim.add_markers(viewer.user_scn, tgt)
                viewer.sync()
            n += 1
            rec["tick_ms"].append((time.perf_counter() - tick0) * 1e3)
            if virtual:
                vt[0] += dt
            elif not a.headless or a.synthetic is False:
                next_t += dt
                slp = next_t - time.monotonic()
                if slp > 0:
                    time.sleep(slp)
                else:
                    next_t = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if a.snapshot and not a.snapshot_every:
            render_snapshot(sim, a.snapshot, tgt)
        if viewer is not None:
            viewer.close()
        wrapper.close()
    summ = _summary(rec, n, wrapper)
    print("[summary] " + json.dumps(summ))
    if a.stats_json:
        Path(a.stats_json).write_text(json.dumps(summ, indent=1))
    return summ


def _snap_name(path, n):
    p = Path(path)
    return str(p.with_name(f"{p.stem}_f{n:04d}{p.suffix}"))


def _fk_err(ik, q, td):
    import pinocchio as pin
    m = ik.reduced_robot.model
    d = m.createData()
    pin.framesForwardKinematics(m, d, q)
    pe, re = 0.0, 0.0
    for f, T in (("L_ee", td.left_wrist_pose), ("R_ee", td.right_wrist_pose)):
        o = d.oMf[m.getFrameId(f)]
        pe = max(pe, float(np.linalg.norm(o.translation - T[:3, 3])))
        re = max(re, float(np.linalg.norm(rot_log(o.rotation.T @ T[:3, :3]))))
    return pe, re


def _summary(rec, n, wrapper):
    def st(x):
        x = np.asarray(x)
        return {"mean": float(x.mean()), "p95": float(np.percentile(x, 95)), "max": float(x.max())} if len(x) else None
    return {"ticks": n, "frames_received": wrapper.stats["frames"], "nan_count": rec["nan"],
            "ik_ms": st(rec["ik_ms"]), "hand_retarget_ms": st(rec["hand_ms"]), "tick_ms": st(rec["tick_ms"]),
            "ik_pos_err_mm": st(rec["pos_err_mm"]), "ik_rot_err_deg": st(rec["rot_err_deg"]),
            "arm_q_abs_max_rad": rec["arm_q_max"],
            "hand_q_range_rad": {s: [np.min(v, 0).round(2).tolist(), np.max(v, 0).round(2).tolist()] for s, v in rec["hq"].items() if v}}


if __name__ == "__main__":
    main()
