"""Pure-numpy frame conversion, relative (clutch) mapping and smoothing for the viz.

No mujoco / zmq imports here so it can be unit-tested anywhere.
"""
from __future__ import annotations

import numpy as np

R_W_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# (parent, child) pairs of the 20 bones, MANO/OpenPose order
BONES = []
for _f in range(5):
    _b = 1 + 4 * _f
    BONES += [(0, _b), (_b, _b + 1), (_b + 1, _b + 2), (_b + 2, _b + 3)]
assert len(BONES) == 20

HANDS = ("left", "right")
# neutral anchor in front of the chest (W), relative to camera height reference
ANCHOR_XY = {"left": (0.35, 0.22), "right": (0.35, -0.22)}
ANCHOR_DZ = -0.30  # below the head


# Robot neutral wrist positions (G1_29_ArmIK FK at q=0, pelvis/waist frame; same constants
# the clutch in xr_camera_teleop/teleop/clutch.py starts from, neutral rotation = identity).
ROBOT_NEUTRAL = {"left": (0.2498, 0.1487, 0.0952), "right": (0.2498, -0.1486, 0.0952)}
WAIST_DROP = 0.50  # default waist reference height = cam_height - WAIST_DROP


def robot_anchors(waist_height):
    """Anchors (W frame) = waist origin placed at waist_height + robot neutral wrist offsets."""
    return {s: np.array([x, y, waist_height + z]) for s, (x, y, z) in ROBOT_NEUTRAL.items()}


def cam_to_world_points(p_cam, cam_height=1.5, scale=1.0):
    p = np.asarray(p_cam, float)
    return scale * (p @ R_W_CAM.T) + np.array([0.0, 0.0, cam_height])


def cam_to_world_rot(R_cam):
    return R_W_CAM @ np.asarray(R_cam, float)


def project_so3(R):
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


class OneEuro:
    """One-Euro filter on an arbitrary ndarray."""

    def __init__(self, min_cutoff=1.5, beta=5.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = self._alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class HandSmoother:
    """Smooths one hand's (wrist_pos, wrist_rot, keypoints) in the camera frame."""

    def __init__(self, **kw):
        self.f = OneEuro(**kw)

    def reset(self):
        self.f.reset()

    def __call__(self, wrist_pos, wrist_rot, kps, t):
        v = np.concatenate([np.ravel(wrist_pos), np.ravel(wrist_rot), np.ravel(kps)])
        v = self.f(v, t)
        return v[:3], project_so3(v[3:12].reshape(3, 3)), v[12:].reshape(21, 3)


def anchor_pose(side, cam_height):
    x, y = ANCHOR_XY[side]
    return np.array([x, y, cam_height + ANCHOR_DZ]), np.eye(3)


class HandMapper:
    """CAM-frame hand dicts -> W-frame (wrist_pos, wrist_rot, keypoints[21,3])."""

    def __init__(self, mode="absolute", scale=None, cam_height=1.5, neutral_rot="ref", anchors=None,
                 shared_anchor=True, max_depth=None):
        assert mode in ("absolute", "relative")
        self.mode = mode
        self.cam_height = cam_height
        self.scale = scale if scale is not None else (1.0 if mode == "absolute" else 0.8)
        self.neutral_rot = neutral_rot  # "ref" (R_R_ref = R_H_ref) or "identity"
        self.anchors = anchors  # optional {side: xyz in W}; overrides the viewer-default anchors
        self.ref = {}  # side -> (p_ref, R_ref) in W without offset (pure rotation of cam)
        self.R_level = np.eye(3)  # camera levelling (teleop/camlevel.py); v_level = R_level @ v_cam
        # egocentric camera: both hands share one anchor (midpoints) so the inter-hand geometry is kept
        self.shared_anchor = shared_anchor
        self.max_depth = max_depth  # ignore hands farther than this from the camera (not the operator)
        self._shared = False

    @property
    def calibrated(self):
        return len(self.ref) > 0

    def _w_pose(self, wrist_pos, wrist_rot):
        # camera-origin W (no height offset): relative mode only uses differences
        return (R_W_CAM @ self.R_level @ np.asarray(wrist_pos, float),
                R_W_CAM @ self.R_level @ np.asarray(wrist_rot, float))

    def calibrate(self, hands):
        """hands: {side: (wrist_pos_cam, wrist_rot_cam)}. Stores refs for the sides given."""
        for side, (p, R) in hands.items():
            self.ref[side] = self._w_pose(p, R)
        self._shared = self.shared_anchor and all(h in hands for h in HANDS)
        if self._shared:
            mid = 0.5 * (self.ref["left"][0] + self.ref["right"][0])
            self.ref = {h: (mid, R) for h, (_, R) in self.ref.items()}

    def clear(self):
        self.ref = {}
        self._shared = False

    def map(self, side, wrist_pos, wrist_rot, kps):
        """Returns (p, R, kps_W) or None if relative and not calibrated for this side."""
        kps = np.asarray(kps, float)
        if self.max_depth and float(np.asarray(wrist_pos)[2]) > self.max_depth:
            return None
        if self.mode == "absolute":
            Rl = self.R_level
            p = cam_to_world_points(Rl @ np.asarray(wrist_pos, float), self.cam_height, self.scale)
            R = cam_to_world_rot(Rl @ np.asarray(wrist_rot, float))
            k = cam_to_world_points(kps @ Rl.T, self.cam_height, self.scale)
            return p, R, k
        if side not in self.ref:
            return None
        p_h, R_h = self._w_pose(wrist_pos, wrist_rot)
        p_ref, R_ref = self.ref[side]
        p_a, R_a = anchor_pose(side, self.cam_height)
        if self.anchors is not None:
            p_a = np.asarray(self.anchors[side], float)
        if self._shared:
            if self.anchors is not None:
                p_a = 0.5 * sum(np.asarray(self.anchors[h], float) for h in HANDS)
            else:
                p_a = 0.5 * (anchor_pose("left", self.cam_height)[0] + anchor_pose("right", self.cam_height)[0])
        if self.neutral_rot == "ref":  # no visual orientation jump at calibration
            R_a = R_ref
        p = p_a + self.scale * (p_h - p_ref)
        R = (R_h @ R_ref.T) @ R_a
        # fingers keep wrist-local shape (unscaled hand size)
        kw = kps @ (R_W_CAM @ self.R_level).T
        local = (kw - p_h) @ R_h  # = R_h^T (k - p_h)
        k = p + local @ R.T
        return p, R, k


def _rotvec(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    a = np.arccos(c)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    n = np.linalg.norm(w)
    return np.zeros(3) if n < 1e-12 else w / n * a


def _expm_so3(v):
    a = np.linalg.norm(v)
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class Extrapolator:
    """Per-hand constant-velocity extrapolation between ~30 Hz samples for a 60 Hz renderer.

    update(t, p, R, k) with the sample time (shared monotonic clock); at(now) predicts
    forward by at most max_lead seconds. Resets if the gap between samples is too large.
    """

    def __init__(self, max_lead=0.05, max_gap=0.25):
        self.max_lead, self.max_gap = max_lead, max_gap
        self.prev = self.cur = None

    def update(self, t, p, R, k):
        if self.cur is not None and t <= self.cur[0]:
            return
        self.prev = self.cur if (self.cur is not None and t - self.cur[0] < self.max_gap) else None
        self.cur = (t, np.asarray(p, float), np.asarray(R, float), np.asarray(k, float))

    def at(self, now):
        t, p, R, k = self.cur
        if self.prev is None:
            return p, R, k
        t0, p0, R0, k0 = self.prev
        lead = min(max(now - t, 0.0), self.max_lead)
        s = lead / (t - t0)
        return (p + s * (p - p0), project_so3(_expm_so3(s * _rotvec(R @ R0.T)) @ R),
                k + s * (k - k0))
