"""Smoothing and rate limiting for teleop targets. Pure numpy.

- OneEuroFilter      : position / generic vector (Casiez et al. 2012)
- SO3OneEuroFilter   : same idea on SO(3) via quaternion slerp, with
                       hemisphere (sign) continuity
- limit_pose_step    : linear / angular velocity limits
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------- SO(3) utils
def mat_to_quat(R):
    """3x3 rotation -> unit quaternion (w,x,y,z), w >= 0 not enforced."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_slerp(q0, q1, a):
    """Slerp from q0 to q1 (a in [0,1]); takes the short arc."""
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    s = np.sin(th)
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / s


def rot_log(R):
    """SO(3) log map -> rotation vector."""
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if np.pi - th < 1e-5:  # near pi: use symmetric part
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        ax = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        # sign from skew part if available
        sk = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(ax, sk) < 0:
            ax = -ax
        return th * ax / np.linalg.norm(ax)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return th / (2 * np.sin(th)) * w


def rot_exp(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def project_so3(R):
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def _alpha(cutoff, dt):
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


# ------------------------------------------------------------------ One-Euro
class OneEuroFilter:
    """One-Euro filter on an ndarray. Call with (x, t_seconds)."""

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or t <= self.t:
            if self.x is None:
                self.x, self.dx = x.copy(), np.zeros_like(x)
            self.t = t if self.t is None else max(t, self.t)
            return self.x.copy()
        dt = t - self.t
        dx = (x - self.x) / dt
        ad = _alpha(self.d_cutoff, dt)
        self.dx = ad * dx + (1 - ad) * self.dx
        speed = np.linalg.norm(self.dx)
        a = _alpha(self.min_cutoff + self.beta * speed, dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()


class SO3OneEuroFilter:
    """One-Euro style filter on SO(3).

    State is a quaternion. Each new sample's sign is flipped to the same
    hemisphere as the state (q and -q are the same rotation), the angular
    speed is low-passed, and the state slerps toward the sample with a
    speed-adaptive alpha. Input/output are 3x3 matrices.
    """

    def __init__(self, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.q = self.t = None
        self.w = 0.0

    def __call__(self, R, t):
        q = mat_to_quat(R)
        if self.q is None:
            self.q, self.t, self.w = q, t, 0.0
            return quat_to_mat(self.q)
        if t <= self.t:
            return quat_to_mat(self.q)
        if np.dot(q, self.q) < 0:      # sign continuity
            q = -q
        dt = t - self.t
        ang = 2 * np.arccos(np.clip(abs(np.dot(q, self.q)), 0.0, 1.0))
        ad = _alpha(self.d_cutoff, dt)
        self.w = ad * (ang / dt) + (1 - ad) * self.w
        a = _alpha(self.min_cutoff + self.beta * self.w, dt)
        self.q = quat_slerp(self.q, q, a)
        self.t = t
        return quat_to_mat(self.q)


class PoseOneEuro:
    """Joint position + rotation One-Euro filter on a 4x4 pose."""

    def __init__(self, pos_min_cutoff=1.5, pos_beta=2.0, rot_min_cutoff=1.5, rot_beta=0.5):
        self.pf = OneEuroFilter(pos_min_cutoff, pos_beta)
        self.rf = SO3OneEuroFilter(rot_min_cutoff, rot_beta)

    def reset(self):
        self.pf.reset()
        self.rf.reset()

    def __call__(self, p, R, t):
        return self.pf(p, t), self.rf(R, t)


# ---------------------------------------------------------------- rate limits
def limit_pose_step(p_prev, R_prev, p_tgt, R_tgt, dt, max_lin=1.2, max_ang=4.0):
    """Move (p_prev,R_prev) toward target obeying linear/angular speed limits."""
    dp = np.asarray(p_tgt, float) - p_prev
    n = np.linalg.norm(dp)
    lim = max_lin * dt
    p = p_tgt.copy() if n <= lim else p_prev + dp * (lim / n)
    dr = rot_log(R_tgt @ R_prev.T)
    a = np.linalg.norm(dr)
    alim = max_ang * dt
    R = R_tgt if a <= alim else rot_exp(dr * (alim / a)) @ R_prev
    return np.asarray(p, float), project_so3(R)


class PoseRateLimiter:
    def __init__(self, max_lin=1.2, max_ang=4.0):
        self.max_lin, self.max_ang = max_lin, max_ang
        self.reset()

    def reset(self, p=None, R=None, t=None):
        self.p, self.R, self.t = p, R, t

    def __call__(self, p, R, t):
        p = np.asarray(p, float)
        if self.p is None:
            self.p, self.R, self.t = p.copy(), np.asarray(R, float).copy(), t
            return self.p.copy(), self.R.copy()
        dt = max(t - self.t, 1e-4)
        self.p, self.R = limit_pose_step(self.p, self.R, p, np.asarray(R, float), dt,
                                         self.max_lin, self.max_ang)
        self.t = t
        return self.p.copy(), self.R.copy()
