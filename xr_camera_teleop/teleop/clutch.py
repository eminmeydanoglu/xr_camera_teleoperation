"""Relative (clutch) mapping from human wrist motion to robot wrist target.

Everything here is expressed in W (z up, x forward, y left); the robot waist
frame used by G1_29_ArmIK has the same axes (robot basis, tv_wrapper.py:6-7),
so no extra axis change is needed between the clutch output and IK input.

    p_robot = p_r0 + k * (p_h - p_h0)
    R_robot = (R_h R_h0^T) R_r0              (delta rotation applied in world)

(p_h0,R_h0) = human wrist at calibration, (p_r0,R_r0) = robot neutral wrist.
Camera placement drops out; only motion is transferred.

Robot neutral choice (see neutral_from_ik_fk / neutral_from_tv_wrapper_const):
  * DEFAULT = FK of G1_29_ArmIK at q = 0 (the model the IK optimises and the
    pose the controller drives to in ctrl_dual_arm_go_home / the IK
    init_data = zeros, robot_arm_ik.py:~182). Starting the clutch there means
    calibration causes *no* arm motion: arm q stays at the home posture.
    FK gives L_ee p=(0.250, +0.149, 0.095), R_ee p=(0.250,-0.149,0.095),
    R = I (palms facing each other, thumbs up; tv_wrapper.py:21-29 URDF
    convention).
  * tv_wrapper's CONST_*_ARM_POSE pipeline gives p=(0.25,+-0.15,0.08) with
    R = Rx(+-90deg) (palm down). Same position within 1.5 cm, different hand
    orientation: it would force a 90 deg wrist roll at calibration, hence
    not default. Available as neutral='tv_const'.
"""
from __future__ import annotations

import numpy as np

from .filters import project_so3, rot_exp, rot_log


def neutral_from_ik_fk(ik, q=None):
    """(T_left, T_right) 4x4 of IK frames L_ee/R_ee in waist frame at config q (default zeros)."""
    import pinocchio as pin

    m = ik.reduced_robot.model
    d = m.createData()
    q = np.zeros(m.nq) if q is None else np.asarray(q, float)
    pin.framesForwardKinematics(m, d, q)
    out = []
    for f in ("L_ee", "R_ee"):
        T = np.eye(4)
        oMf = d.oMf[m.getFrameId(f)]
        T[:3, :3] = oMf.rotation
        T[:3, 3] = oMf.translation
        out.append(T)
    return out[0], out[1]


def neutral_from_tv_wrapper_const():
    """Waist-frame wrist poses tv_wrapper yields for its CONST_*_ARM_POSE / CONST_HEAD_POSE.

    Replicates tv_wrapper.py:155-169 (constants), :137-153 (basis change),
    :122-130 (arm convention), :103-119 (waist offset, +0.15 x, +0.45 z).
    """
    T_ROBOT_OPENXR = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    T_OPENXR_ROBOT = T_ROBOT_OPENXR.T.copy()
    T_OPENXR_ROBOT[:3, 3] = -T_ROBOT_OPENXR[:3, :3].T @ T_ROBOT_OPENXR[:3, 3]
    head = np.array([[1, 0, 0, 0], [0, 1, 0, 1.5], [0, 0, 1, -0.2], [0, 0, 0, 1.0]])
    la = np.array([[1, 0, 0, -0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1.0]])
    ra = np.array([[1, 0, 0, 0.15], [0, 1, 0, 1.13], [0, 0, 1, -0.3], [0, 0, 0, 1.0]])
    TL = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    TR = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1.0]])
    bh = T_ROBOT_OPENXR @ head @ T_OPENXR_ROBOT
    res = []
    for a, T in ((la, TL), (ra, TR)):
        w = T_ROBOT_OPENXR @ a @ T_OPENXR_ROBOT @ T
        w[:3, 3] = w[:3, 3] - bh[:3, 3]   # head_yaw w/ identity head yaw
        w[0, 3] += 0.15
        w[2, 3] += 0.45
        res.append(w)
    return res[0], res[1]


class HandClutch:
    """Per-hand relative mapping with hold-last-valid and staleness handling."""

    def __init__(self, neutral_T, k=0.8, stale_timeout=1.0, max_offset=0.40,
                 reanchor_on_stale=True, return_lin=0.15, return_ang=0.8, R_offset=None):
        self.R_offset = None if R_offset is None else np.asarray(R_offset, float)  # absolute-rotation mode
        self.neutral_T = np.asarray(neutral_T, float).copy()
        self.k = np.broadcast_to(np.asarray(k, float), (3,)).copy()
        self.stale_timeout = stale_timeout
        self.max_offset = max_offset
        self.reanchor_on_stale = reanchor_on_stale
        self.return_lin, self.return_ang = return_lin, return_ang   # m/s, rad/s back to neutral when stale
        self.reset()

    # -- state ---------------------------------------------------------
    def reset(self):
        """Back to uncalibrated: output is the robot neutral."""
        self.calibrated = False
        self._pending = False
        self.p_h0 = self.R_h0 = None
        self.p_r0 = self.neutral_T[:3, 3].copy()
        self.R_r0 = self.neutral_T[:3, :3].copy()
        self.last_T = self.neutral_T.copy()
        self.last_valid_t = None
        self.last_p = self.last_R = None
        self.stale = False
        self.engaged = True
        self._reanchor_pending = False
        self._last_t = None

    def set_engaged(self, engaged: bool):
        """Clutch pedal: while disengaged the output is frozen; on re-engage the
        current human pose is re-anchored to the frozen output (no jump)."""
        engaged = bool(engaged)
        if engaged and not self.engaged:
            self._reanchor_pending = True
        self.engaged = engaged

    def request_calibration(self):
        """Calibrate on the next valid sample (or immediately from the last one)."""
        self._pending = True

    def calibrate(self, p_h, R_h, t=None, p_h_anchor=None, p_r_anchor=None):
        """Anchor now: human pose (p_h,R_h) <-> robot neutral.

        Shared (egocentric) anchor: pass p_h_anchor = midpoint of both human wrists and
        p_r_anchor = midpoint of both robot neutrals, so the inter-hand geometry is kept
        (hands together -> robot hands together) instead of each hand snapping to its neutral."""
        self.p_h0 = np.asarray(p_h if p_h_anchor is None else p_h_anchor, float).copy()
        self.R_h0 = project_so3(np.asarray(R_h, float))
        self.p_r0 = (self.neutral_T[:3, 3] if p_r_anchor is None else np.asarray(p_r_anchor, float)).copy()
        self.R_r0 = self.neutral_T[:3, :3].copy()
        self.calibrated = True
        self._pending = False
        self.stale = False
        if t is not None:
            self.last_valid_t = t

    def _anchor_to_output(self, p_h, R_h):
        """Re-clutch: current human pose <-> last robot output (no jump)."""
        self.p_h0 = np.asarray(p_h, float).copy()
        self.R_h0 = project_so3(np.asarray(R_h, float))
        self.p_r0 = self.last_T[:3, 3].copy()
        self.R_r0 = self.last_T[:3, :3].copy()

    # -- mapping -------------------------------------------------------
    def map_pose(self, p_h, R_h):
        p = self.p_r0 + self.k * (np.asarray(p_h) - self.p_h0)
        off = p - self.neutral_T[:3, 3]
        n = np.linalg.norm(off)
        if self.max_offset and n > self.max_offset:     # keep targets reachable
            p = self.neutral_T[:3, 3] + off * (self.max_offset / n)
        if self.R_offset is not None:       # absolute: human wrist frame (W) -> robot ee convention
            R = project_so3(np.asarray(R_h) @ self.R_offset)
        else:
            R = project_so3((np.asarray(R_h) @ self.R_h0.T) @ self.R_r0)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        return T

    def _step_to_neutral(self, dt):
        """Rate-limited drift of the held output back to the robot neutral (hand lost > stale_timeout)."""
        T = self.last_T
        dp = self.neutral_T[:3, 3] - T[:3, 3]
        n = np.linalg.norm(dp)
        if n > 1e-9:
            T[:3, 3] += dp * min(1.0, self.return_lin * dt / n)
        w = rot_log(self.neutral_T[:3, :3] @ T[:3, :3].T)       # world-frame error rotation
        a = np.linalg.norm(w)
        if a > 1e-9:
            T[:3, :3] = project_so3(rot_exp(w * min(1.0, self.return_ang * dt / a)) @ T[:3, :3])

    def hold_neutral(self, t):
        """Forced return to the robot neutral (e.g. handedness conflict): like a stale hand, but
        immediately and regardless of calibration. Returns the 4x4 output."""
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        self.stale = True
        if self.engaged and dt > 0:
            self._step_to_neutral(min(dt, 0.2))
        return self.last_T.copy()

    def update(self, t, p_h=None, R_h=None, valid=True):
        """Feed one tick. Returns (4x4 robot wrist target, info dict)."""
        have = valid and p_h is not None and R_h is not None
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if have and not self.engaged:
            self.last_p, self.last_R = np.asarray(p_h, float), np.asarray(R_h, float)
            self.last_valid_t = t
            self.stale = False
            return self.last_T.copy(), {"calibrated": self.calibrated, "stale": False,
                                        "tracking": True, "engaged": False}
        if have:
            self.last_p, self.last_R = np.asarray(p_h, float), np.asarray(R_h, float)
            was_stale = (self.last_valid_t is not None
                         and t - self.last_valid_t > self.stale_timeout)
            if self._pending:
                self.calibrate(p_h, R_h)
            elif self.calibrated and (self._reanchor_pending
                                      or (was_stale and self.reanchor_on_stale)):
                self._anchor_to_output(p_h, R_h)
            self._reanchor_pending = False
            self.last_valid_t = t
            self.stale = False
            if self.calibrated:
                self.last_T = self.map_pose(p_h, R_h)
        else:
            if self.last_valid_t is not None and t - self.last_valid_t > self.stale_timeout:
                self.stale = True
            elif self.last_valid_t is None:
                self.stale = True
            if self.stale and self.calibrated and self.engaged and dt > 0:
                self._step_to_neutral(min(dt, 0.2))
        return self.last_T.copy(), {"calibrated": self.calibrated, "stale": self.stale,
                                    "tracking": bool(have), "engaged": self.engaged}


class StillDetector:
    """True once the tracked point(s) stayed within `radius` (m) of their window mean for `window`
    seconds continuously (~ wrist speed < 2*radius/window = 6 cm/s). Feed push(side,t,p) on each
    detection and lost(side) when the hand is absent; ready() = every side in `sides` still."""

    def __init__(self, sides=("left", "right"), window=1.0, radius=0.03, max_gap=0.25):
        self.sides, self.window, self.radius, self.max_gap = tuple(sides), window, radius, max_gap
        self.buf = {s: [] for s in self.sides}

    def reset(self):
        self.buf = {s: [] for s in self.sides}

    def lost(self, side):
        self.buf[side] = []

    def push(self, side, t, p):
        b = self.buf[side]
        if b and t - b[-1][0] > self.max_gap:
            b.clear()
        b.append((t, np.asarray(p, float).copy()))
        while len(b) > 2 and t - b[1][0] >= self.window:     # keep one sample just outside the window
            b.pop(0)

    def still(self, side):
        b = self.buf[side]
        if len(b) < 3 or b[-1][0] - b[0][0] < self.window:
            return False
        P = np.array([p for _, p in b])
        return bool(np.max(np.linalg.norm(P - P.mean(0), axis=1)) < self.radius)

    def ready(self):
        return all(self.still(s) for s in self.sides)
