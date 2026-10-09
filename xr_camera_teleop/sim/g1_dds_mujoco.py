#!/usr/bin/env python
"""MuJoCo Unitree G1 (29 DoF + Dex3-1) as a DDS stand-in for the real robot.

Speaks exactly what xr_teleoperate's G1_29_ArmController / Dex3_1_Controller speak:

  subscribes  rt/lowcmd        hg LowCmd_   (debug mode: all 29 motors)
              rt/arm_sdk       hg LowCmd_   (--motion: waist+arms, motor_cmd[29].q = weight)
              rt/dex3/{left,right}/cmd   HandCmd_
              rt/reset_pose/cmd          String_ (scene reset, used by teleop_hand_and_arm --sim)
  publishes   rt/lowstate      hg LowState_ @ 500 Hz (q, dq, ddq, tau_est, IMU, mode_machine, CRC)
              rt/dex3/{left,right}/state HandState_ @ 100 Hz
              rt/sim_state     String_ JSON @ 10 Hz (minimal)

Motor model = hg LowCmd semantics:  tau = kp (q* - q) + kd (dq* - dq) + tau_ff, clipped to the
joint's actuator force range; motor mode 0 => no torque.

rt/arm_sdk (the "--motion" path): on the real G1 the closed-source Unitree locomotion/balance
controller keeps running and arm_sdk is blended with it by the weight w = motor_cmd[29].q.
That controller is not available here, so the "internal controller" is a PD hold of the
default standing posture (legs/waist/arms at q=0): tau = w * tau_cmd + (1-w) * tau_hold on waist+arm
joints, legs always tau_hold. Same fallback when lowcmd/arm_sdk go stale (>0.25 s). The pelvis is
welded at standing height (default) so there is no balance problem; --elastic-band instead leaves
the base free and hangs it from a virtual spring (legs only hold PD: it will not walk or balance).

Real-time: physics steps are locked to the wall clock (dt = 1 ms default); a 60 Hz mujoco passive
viewer is synced from the same loop (no offscreen rendering, no image streaming); DDS callbacks
only swap a reference to the newest message (never touch physics); GC is frozen/disabled.
RTF and per-step time percentiles are printed every --report seconds.
--report-self-collision: every 50 ms the arm q is checked for self-collision on the URDF collision
meshes (xr_camera_teleop/teleop/self_collision.MeshGate, ~0.2 ms); contacts are REPORTED (not blocked: the
MJCF geoms have contype=0) - rate-limited line on contact + counts in the periodic report.

    data/xr-env/bin/python xr_camera_teleop/sim/g1_dds_mujoco.py [--headless] [--domain 1 --iface lo]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
G1_XML = REPO / "third_party" / "xr_teleoperate" / "assets" / "g1" / "g1_body29_hand14.xml"

BODY_JOINTS = [  # index == G1_29_JointIndex (robot_arm.py)
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]
BODY_JOINTS = [n + "_joint" for n in BODY_JOINTS]
# Dex3_1_{Left,Right}_JointIndex (robot_hand_unitree.py): note index/middle order differs per side
HAND_JOINTS = {
    "left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
    "right": ["thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"],
}
HAND_JOINTS = {s: [f"{s}_hand_{n}_joint" for n in v] for s, v in HAND_JOINTS.items()}
N_BODY, N_HAND, N_MOTORS = 29, 7, 35
WAIST_ARMS = slice(12, 29)
K_NOT_USED_0 = 29            # arm_sdk weight slot

# rotor armature [kg m^2] (reflected, from Unitree 5020/7520/4010 actuators, as in unitree_rl_gym / mjlab)
A5020, A7520_14, A7520_22, A4010 = 0.003609725, 0.010177520, 0.025101925, 0.00425
A_HAND = 5e-4                # keeps explicit hand PD (kd=0.2) stable at dt=1ms


def armature_for(name: str) -> float:
    if "_hand_" in name:
        return A_HAND
    if any(k in name for k in ("hip_pitch", "hip_yaw", "waist_yaw")):
        return A7520_14
    if any(k in name for k in ("hip_roll", "knee")):
        return A7520_22
    if any(k in name for k in ("ankle", "waist_roll", "waist_pitch")):
        return 2 * A5020
    if any(k in name for k in ("wrist_pitch", "wrist_yaw")):
        return A4010
    return A5020


# hold gains of the stand-in "internal controller" (legs / unblended joints)
HOLD_KP = np.array([150, 150, 150, 200, 40, 40] * 2 + [200, 200, 200] + [60] * 4 + [20] * 3 + [60] * 4 + [20] * 3, float)
HOLD_KD = np.array([4, 4, 4, 5, 2, 2] * 2 + [5, 5, 5] + [3] * 4 + [1.5] * 3 + [3] * 4 + [1.5] * 3, float)
HAND_HOLD_KP, HAND_HOLD_KD = 1.5, 0.2


class Cmd:
    """Immutable-by-convention parsed command; the DDS thread replaces the reference atomically."""
    __slots__ = ("t", "mode", "q", "dq", "tau", "kp", "kd", "t_rx")

    def __init__(self, motor_cmd, n, t):
        self.t = t
        self.mode = np.fromiter((m.mode for m in motor_cmd[:n]), np.int32, n)
        self.q = np.fromiter((m.q for m in motor_cmd[:n]), np.float64, n)
        self.dq = np.fromiter((m.dq for m in motor_cmd[:n]), np.float64, n)
        self.tau = np.fromiter((m.tau for m in motor_cmd[:n]), np.float64, n)
        self.kp = np.fromiter((m.kp for m in motor_cmd[:n]), np.float64, n)
        self.kd = np.fromiter((m.kd for m in motor_cmd[:n]), np.float64, n)


def build_model(dt: float, fixed_base: bool):
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.option.timestep = dt
    if fixed_base:
        for j in list(spec.body("pelvis").joints):      # weld pelvis at standing height
            spec.delete(j)
    for j in spec.joints:
        if j.type != mujoco.mjtJoint.mjJNT_FREE:
            j.armature = armature_for(j.name)
    return spec.compile()


class G1DdsSim:
    def __init__(self, a):
        self.a = a
        self.dt = a.dt
        self.model = build_model(a.dt, fixed_base=not a.elastic_band)
        self.data = mujoco.MjData(self.model)
        m = self.model
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        names = BODY_JOINTS + HAND_JOINTS["left"] + HAND_JOINTS["right"]
        self.jids = np.array([jid(n) for n in names])
        self.qadr = m.jnt_qposadr[self.jids]
        self.dadr = m.jnt_dofadr[self.jids]
        self.act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names])
        assert (self.act >= 0).all() and (self.jids >= 0).all()
        lim = m.jnt_actfrcrange[self.jids]
        self.tau_lo, self.tau_hi = lim[:, 0].copy(), lim[:, 1].copy()
        self.n_hand_off = N_BODY
        self.hand_sl = {"left": slice(N_BODY, N_BODY + N_HAND), "right": slice(N_BODY + N_HAND, N_BODY + 2 * N_HAND)}
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.sid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, n) for n in
                    ("imu-pelvis-angular-velocity", "imu-pelvis-linear-acceleration")}
        self.reset()

        # latest-only command buffers (reference swap is atomic in CPython)
        self.low = None
        self.armsdk = None
        self.hand = {"left": None, "right": None}
        self.reset_flag = False
        self.stats = {"low_rx": 0, "arm_rx": 0, "hand_rx_l": 0, "hand_rx_r": 0, "crc_bad": 0,
                      "lowstate_tx": 0, "hand_tx": 0}
        self.trace_cmd = [] if a.trace else None     # (t_rx, source, q_cmd[29])
        self.trace_state = [] if a.trace else None   # (t, q[29], dq[29])
        self.hand_trace = [] if a.trace else None
        self.snap = None
        self.snap_ev = threading.Event()
        self.running = True
        self.sim_t = 0.0
        self.nan_events = 0
        self.sc_gate = None
        self.sc = {"checks": 0, "hits": 0, "last_print": -1e9}
        if a.report_self_collision:
            sys.path.insert(0, str(REPO))
            from xr_camera_teleop.teleop.self_collision import MeshGate
            self.sc_gate = MeshGate()
            arm_names = list(self.sc_gate.model.names[1:])
            assert arm_names == BODY_JOINTS[15:29], arm_names
            self.sc_qadr = self.qadr[15:29]

    # --------------------------------------------------------------------------- state
    def reset(self):
        d = self.data
        mujoco.mj_resetData(self.model, d)
        mujoco.mj_forward(self.model, d)
        self.band_anchor = d.xpos[self.pelvis].copy() + np.array([0, 0, 0.06])

    def read_q(self):
        d = self.data
        return d.qpos[self.qadr], d.qvel[self.dadr]

    # --------------------------------------------------------------------------- DDS
    def init_dds(self):
        from unitree_sdk2py.core.channel import (ChannelFactoryInitialize, ChannelPublisher,
                                                 ChannelSubscriber)
        from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__HandState_,
                                                unitree_hg_msg_dds__LowState_)
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as hg_LowCmd
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
        from unitree_sdk2py.utils.crc import CRC

        a = self.a
        ChannelFactoryInitialize(a.domain, a.iface)
        self.crc = CRC()
        self.String_ = String_

        def on_low(msg):
            if a.check_crc and msg.crc != self.crc.Crc(msg):
                self.stats["crc_bad"] += 1
                return
            c = Cmd(msg.motor_cmd, N_MOTORS, time.monotonic())
            self.stats["low_rx"] += 1
            self.low = c
            if self.trace_cmd is not None:
                self.trace_cmd.append((c.t, 0, c.q[:N_BODY].copy()))

        def on_arm(msg):
            if a.check_crc and msg.crc != self.crc.Crc(msg):
                self.stats["crc_bad"] += 1
                return
            c = Cmd(msg.motor_cmd, N_MOTORS, time.monotonic())
            self.stats["arm_rx"] += 1
            self.armsdk = c
            if self.trace_cmd is not None:
                self.trace_cmd.append((c.t, 1, c.q[:N_BODY].copy()))

        def make_hand_cb(side):
            def cb(msg):
                c = Cmd(msg.motor_cmd, N_HAND, time.monotonic())
                # status bits 4..6 of motor_mode: 0 => motor disabled
                c.mode = (c.mode >> 4) & 0x7
                self.stats["hand_rx_" + side[0]] += 1
                self.hand[side] = c
                if self.hand_trace is not None and side == "left":
                    self.hand_trace.append((c.t, c.q.copy()))
            return cb

        def on_reset(msg):
            self.reset_flag = True
            print(f"[sim] reset requested (category {msg.data})", flush=True)

        self._subs = []
        for topic, typ, cb in (("rt/lowcmd", hg_LowCmd, on_low), ("rt/arm_sdk", hg_LowCmd, on_arm),
                               ("rt/dex3/left/cmd", HandCmd_, make_hand_cb("left")),
                               ("rt/dex3/right/cmd", HandCmd_, make_hand_cb("right")),
                               ("rt/reset_pose/cmd", String_, on_reset)):
            s = ChannelSubscriber(topic, typ)
            s.Init(cb, 0)
            self._subs.append(s)
        self.pub_low = ChannelPublisher("rt/lowstate", hg_LowState)
        self.pub_low.Init()
        self.pub_hand = {s: ChannelPublisher(f"rt/dex3/{s}/state", HandState_) for s in ("left", "right")}
        for p in self.pub_hand.values():
            p.Init()
        self.pub_sim = ChannelPublisher("rt/sim_state", String_)
        self.pub_sim.Init()
        self.low_msg = unitree_hg_msg_dds__LowState_()
        self.low_msg.mode_machine = a.mode_machine
        self.hand_msg = {s: unitree_hg_msg_dds__HandState_() for s in ("left", "right")}
        self.pub_thread = threading.Thread(target=self._pub_loop, daemon=True, name="dds_pub")
        self.pub_thread.start()

    def _pub_loop(self):
        """Publishes snapshots handed over by the physics loop (never blocks it)."""
        n = 0
        while self.running:
            if not self.snap_ev.wait(0.1):
                continue
            self.snap_ev.clear()
            s = self.snap
            if s is None:
                continue
            tick, q, dq, ddq, tau, quat, gyro, acc = s
            m = self.low_msg
            m.tick = tick
            im = m.imu_state
            im.quaternion = quat.tolist()
            im.gyroscope = gyro.tolist()
            im.accelerometer = acc.tolist()
            w, x, y, z = quat
            im.rpy = [float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
                      float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))),
                      float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))]
            ql, dql, ddl, tl = q[:N_BODY].tolist(), dq[:N_BODY].tolist(), ddq[:N_BODY].tolist(), tau[:N_BODY].tolist()
            for i in range(N_BODY):
                ms = m.motor_state[i]
                ms.mode = 1
                ms.q, ms.dq, ms.ddq, ms.tau_est = ql[i], dql[i], ddl[i], tl[i]
            m.crc = self.crc.Crc(m)
            self.pub_low.Write(m)
            self.stats["lowstate_tx"] += 1
            n += 1
            if n % self.hand_every == 0:
                for side, sl in self.hand_sl.items():
                    hm = self.hand_msg[side]
                    qs, ds, ts = q[sl].tolist(), dq[sl].tolist(), tau[sl].tolist()
                    for i in range(N_HAND):
                        ms = hm.motor_state[i]
                        # +1e-5 rad "encoder bias": Dex3_1_Controller waits for a non-zero state
                        ms.mode, ms.q, ms.dq, ms.tau_est = 1, qs[i] + 1e-5, ds[i], ts[i]
                    self.pub_hand[side].Write(hm)
                self.stats["hand_tx"] += 1
            if n % 50 == 0:
                self.pub_sim.Write(self.String_(data=json.dumps(
                    {"sim_time": round(self.sim_t, 4), "tick": tick, "source": "g1_dds_mujoco"})))

    # --------------------------------------------------------------------------- control
    def compute_tau(self, now, q, dq):
        """hg LowCmd semantics for all 43 DoF (29 body + 14 hand). Returns clipped torque (43,)."""
        tau = np.empty(N_BODY + 2 * N_HAND)
        tau_hold = HOLD_KP * (0.0 - q[:N_BODY]) - HOLD_KD * dq[:N_BODY]
        low, arm = self.low, self.armsdk
        used = "hold"
        if low is not None and now - low.t < self.a.cmd_timeout:
            tb = np.where(low.mode[:N_BODY] != 0,
                          low.kp[:N_BODY] * (low.q[:N_BODY] - q[:N_BODY]) +
                          low.kd[:N_BODY] * (low.dq[:N_BODY] - dq[:N_BODY]) + low.tau[:N_BODY], 0.0)
            tau[:N_BODY] = tb
            used = "lowcmd"
        elif arm is not None and now - arm.t < self.a.cmd_timeout:
            w = float(np.clip(arm.q[K_NOT_USED_0], 0.0, 1.0))
            tc = np.where(arm.mode[:N_BODY] != 0,
                          arm.kp[:N_BODY] * (arm.q[:N_BODY] - q[:N_BODY]) +
                          arm.kd[:N_BODY] * (arm.dq[:N_BODY] - dq[:N_BODY]) + arm.tau[:N_BODY], 0.0)
            tau[:N_BODY] = tau_hold
            tau[WAIST_ARMS] = w * tc[WAIST_ARMS] + (1.0 - w) * tau_hold[WAIST_ARMS]
            used = "arm_sdk"
        else:
            tau[:N_BODY] = tau_hold
        for side, sl in self.hand_sl.items():
            h = self.hand[side]
            if h is not None and now - h.t < self.a.cmd_timeout:
                th = np.where(h.mode != 0, h.kp * (h.q - q[sl]) + h.kd * (h.dq - dq[sl]) + h.tau, 0.0)
            else:
                th = HAND_HOLD_KP * (0.0 - q[sl]) - HAND_HOLD_KD * dq[sl]
            tau[sl] = th
        np.clip(tau, self.tau_lo, self.tau_hi, out=tau)
        self.cmd_source = used
        return tau

    def apply_band(self):
        d = self.data
        p, v = d.xpos[self.pelvis], d.cvel[self.pelvis][3:6]
        f = 8000.0 * (self.band_anchor - p) - 400.0 * v
        # keep upright: torque on pelvis proportional to tilt (gravity vector in body frame)
        R = d.xmat[self.pelvis].reshape(3, 3)
        tilt = np.cross(R[:, 2], np.array([0.0, 0.0, 1.0]))
        tq = 300.0 * tilt - 30.0 * d.cvel[self.pelvis][0:3]
        d.xfrc_applied[self.pelvis, :3] = f
        d.xfrc_applied[self.pelvis, 3:] = tq

    def snapshot(self, tick):
        d, m = self.data, self.model
        q, dq = d.qpos[self.qadr], d.qvel[self.dadr]
        ddq = d.qacc[self.dadr]
        tau = d.ctrl[self.act]
        quat = d.xquat[self.pelvis]
        sa = lambda n: d.sensordata[m.sensor_adr[self.sid[n]]:m.sensor_adr[self.sid[n]] + 3]
        self.snap = (tick, q.copy(), dq.copy(), ddq.copy(), tau.copy(), quat.copy(),
                     sa("imu-pelvis-angular-velocity").copy(), sa("imu-pelvis-linear-acceleration").copy())
        if self.trace_state is not None:
            self.trace_state.append((time.monotonic(), q[:N_BODY].copy(), dq[:N_BODY].copy(), q[N_BODY:].copy()))
        self.snap_ev.set()

    # --------------------------------------------------------------------------- main loop
    def run(self):
        a, m, d = self.a, self.model, self.data
        self.hand_every = max(1, int(round(500.0 / a.hand_hz)))
        self.pub_every = max(1, int(round(1.0 / (a.lowstate_hz * self.dt))))
        if not a.no_dds:
            self.init_dds()
        viewer = None
        if not a.headless:
            from mujoco import viewer as mjviewer
            viewer = mjviewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False)
            viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 150, -12
        view_every = max(1, int(round(1.0 / (a.view_hz * self.dt))))
        sc_every = max(1, int(round(0.05 / self.dt)))

        gc.collect()
        gc.freeze()
        gc.disable()
        step_times = np.zeros(int(a.report / self.dt) + 64)
        ns = 0
        step = 0
        t0 = time.perf_counter()
        next_t = t0
        win_wall, win_sim = t0, 0.0
        last_gc = t0
        overruns = 0
        print(f"[sim] running: dt={self.dt*1e3:.2f} ms, lowstate {a.lowstate_hz:.0f} Hz, hands {a.hand_hz:.0f} Hz, "
              f"base={'elastic band' if a.elastic_band else 'fixed'}, domain={a.domain} iface={a.iface}, "
              f"viewer={'off' if a.headless else '60 Hz passive'}", flush=True)
        run_until = None if a.duration <= 0 else t0 + a.duration
        while self.running:
            ts = time.perf_counter()
            if self.reset_flag:
                self.reset_flag = False
                self.low = self.armsdk = None
                self.hand = {"left": None, "right": None}
                self.reset()
            now = time.monotonic()
            q, dq = self.read_q()
            tau = self.compute_tau(now, q, dq)
            d.ctrl[self.act] = tau
            if a.elastic_band:
                self.apply_band()
            mujoco.mj_step(m, d)
            step += 1
            self.sim_t = d.time
            if step % self.pub_every == 0 and not a.no_dds:
                self.snapshot(int(d.time * 1000))
            elif step % self.pub_every == 0 and self.trace_state is not None:
                self.snapshot(0)
            if viewer is not None and step % view_every == 0:
                if not viewer.is_running():
                    break
                viewer.sync()
            if self.sc_gate is not None and step % sc_every == 0:
                self.check_self_collision()
            te = time.perf_counter()
            if ns < len(step_times):
                step_times[ns] = te - ts
                ns += 1

            # report
            if te - win_wall >= a.report:
                rtf = (d.time - win_sim) / (te - win_wall)
                st = step_times[:ns] * 1e6
                bad = not np.isfinite(d.qpos).all()
                if bad:
                    self.nan_events += 1
                    print("[sim] NaN in qpos -> reset", flush=True)
                    self.reset()
                print(f"[sim] t={d.time:7.2f}s RTF={rtf:.3f}  step us p50={np.percentile(st,50):.0f} "
                      f"p99={np.percentile(st,99):.0f} max={st.max():.0f}  overruns={overruns}  src={self.cmd_source}  "
                      f"rx low/arm/handL/handR={self.stats['low_rx']}/{self.stats['arm_rx']}/"
                      f"{self.stats['hand_rx_l']}/{self.stats['hand_rx_r']} crc_bad={self.stats['crc_bad']} "
                      f"tx low/hand={self.stats['lowstate_tx']}/{self.stats['hand_tx']}"
                      + (f"  self-collision {self.sc['hits']}/{self.sc['checks']} checks" if self.sc_gate else ""),
                      flush=True)
                self._last_report = dict(rtf=rtf, p50=float(np.percentile(st, 50)), p99=float(np.percentile(st, 99)),
                                         max=float(st.max()))
                self.reports.append(self._last_report)
                win_wall, win_sim, ns = te, d.time, 0

            if te - last_gc > 5.0:           # cheap young-generation collect during slack
                gc.collect(0)
                last_gc = te
            if run_until is not None and te > run_until:
                break

            # wall-clock lock: sleep most of the slack, spin the last ~150 us
            next_t += self.dt
            slack = next_t - time.perf_counter()
            if slack > 3e-4:
                time.sleep(slack - 2e-4)
            elif slack < -0.05:
                overruns += 1
                next_t = time.perf_counter()
            while time.perf_counter() < next_t:
                pass

        self.running = False
        if viewer is not None:
            viewer.close()
        self.finish()

    reports: list = []

    def check_self_collision(self):
        q = self.data.qpos[self.sc_qadr]
        hits = self.sc_gate.colliding_pairs(q)
        self.sc["checks"] += 1
        if hits:
            self.sc["hits"] += 1
            now = time.monotonic()
            if now - self.sc["last_print"] >= 1.0:
                self.sc["last_print"] = now
                print(f"[sim] SELF-COLLISION t={self.data.time:.2f}s: {hits[:4]}", flush=True)

    def finish(self):
        a = self.a
        if a.trace:
            cmd = self.trace_cmd
            np.savez(a.trace,
                     cmd_t=np.array([c[0] for c in cmd]), cmd_src=np.array([c[1] for c in cmd]),
                     cmd_q=np.array([c[2] for c in cmd]),
                     st_t=np.array([s[0] for s in self.trace_state]), st_q=np.array([s[1] for s in self.trace_state]),
                     st_dq=np.array([s[2] for s in self.trace_state]), st_hand=np.array([s[3] for s in self.trace_state]),
                     hand_cmd_t=np.array([h[0] for h in self.hand_trace]), hand_cmd_q=np.array([h[1] for h in self.hand_trace]))
            print(f"[sim] trace written to {a.trace}", flush=True)
        if a.summary:
            rep = self.reports[1:] or self.reports       # skip warm-up window
            out = {"rtf_mean": float(np.mean([r["rtf"] for r in rep])) if rep else None,
                   "rtf_min": float(np.min([r["rtf"] for r in rep])) if rep else None,
                   "rtf_max": float(np.max([r["rtf"] for r in rep])) if rep else None,
                   "step_p50_us": float(np.mean([r["p50"] for r in rep])) if rep else None,
                   "step_p99_us": float(np.max([r["p99"] for r in rep])) if rep else None,
                   "step_max_us": float(np.max([r["max"] for r in rep])) if rep else None,
                   "nan_events": self.nan_events, "stats": self.stats,
                   "self_collision": ({"checks": self.sc["checks"], "hits": self.sc["hits"]}
                                      if self.sc_gate is not None else None),
                   "q_final": self.data.qpos[self.qadr].tolist()}
            Path(a.summary).write_text(json.dumps(out, indent=1))
            print(f"[sim] summary written to {a.summary}", flush=True)

    def snapshot_png(self, path, w=960, h=720, azimuth=150, elevation=-10, distance=2.6):
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, w)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, h)
        r = mujoco.Renderer(self.model, h, w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.95]
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r.update_scene(self.data, cam)
        import cv2
        cv2.imwrite(path, cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        r.close()


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", type=int, default=1, help="DDS domain id (teleop --sim => 1, real robot => 0)")
    p.add_argument("--iface", default="lo", help="network interface for DDS (loopback by default)")
    p.add_argument("--dt", type=float, default=0.001, help="physics step [s] (0.0005..0.001)")
    p.add_argument("--headless", action="store_true", help="no viewer window")
    p.add_argument("--elastic-band", action="store_true", help="free base hung from a virtual band instead of welded pelvis")
    p.add_argument("--lowstate-hz", type=float, default=500.0)
    p.add_argument("--hand-hz", type=float, default=100.0)
    p.add_argument("--view-hz", type=float, default=60.0)
    p.add_argument("--cmd-timeout", type=float, default=0.25, help="s without lowcmd/arm_sdk before falling back to PD hold")
    p.add_argument("--mode-machine", type=int, default=5, help="LowState mode_machine (G1 29dof rev1.0 = 5)")
    p.add_argument("--no-check-crc", dest="check_crc", action="store_false", help="accept commands with bad CRC")
    p.add_argument("--no-dds", action="store_true", help="debug: no DDS at all")
    p.add_argument("--report", type=float, default=5.0, help="RTF / step-time report interval [s]")
    p.add_argument("--report-self-collision", action="store_true",
                   help="check arm self-collision on the URDF meshes every 50 ms and report it (not blocked)")
    p.add_argument("--duration", type=float, default=0.0, help="stop after N wall seconds (0 = until Ctrl-C / window closed)")
    p.add_argument("--trace", default="", help="write npz with command/state traces on exit (latency analysis)")
    p.add_argument("--summary", default="", help="write json with RTF/step-time summary on exit")
    p.add_argument("--snapshot", default="", help="on exit save a PNG with mujoco.Renderer (needs MUJOCO_GL=egl headless)")
    return p.parse_args()


def main():
    a = parse()
    sim = G1DdsSim(a)
    sim.reports = []
    signal.signal(signal.SIGINT, lambda *_: setattr(sim, "running", False))
    signal.signal(signal.SIGTERM, lambda *_: setattr(sim, "running", False))
    try:
        sim.run()
    finally:
        if a.snapshot:
            gc.enable()
            sim.snapshot_png(a.snapshot)
            print(f"[sim] snapshot {a.snapshot}", flush=True)
        os._exit(0)      # cyclonedds listener threads can block interpreter shutdown


if __name__ == "__main__":
    main()
