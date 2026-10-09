"""Self-collision-aware drop-in replacement for xr_teleoperate's G1_29_ArmIK (third_party untouched).

SafeG1_29_ArmIK subclasses the upstream class: the upstream __init__ builds the same casadi.Opti
problem (50*pos^2 + rot^2 + 0.02|q|^2 + 0.1|q-q_last|^2, joint limits), then we

  layer 1: build a second Opti problem = same cost + the sphere/capsule self-collision constraints of
           self_collision.SphereModel (arm vs torso/head/pelvis/thighs, left arm vs right arm; 793 pairs
           soft-min-aggregated into 58 smooth IPOPT inequalities), so a target inside the body makes the
           arm stop at the surface. Fast path: the upstream (unconstrained) problem is solved first; when
           its optimum satisfies the constraints it is also the constrained optimum and is used as is
           (~1 ms); otherwise the constrained problem is solved (~10 ms);
  layer 2: check every output (after upstream's WeightedMovingFilter) on the real URDF collision
           meshes (self_collision.MeshGate, hpp-fcl). A colliding or non-finite q is never returned:
           the last safe q and its tauff are returned instead, with a rate-limited stderr warning.

solve_ik() keeps the upstream signature and return value (sol_q[14], sol_tauff[14]). Differences
to upstream on a solver failure (max_iter / infeasible): upstream returns the current motor q and a
zero tauff (None if no q was passed) and pushes the failed iterate into its filter; we return the
current motor q (if collision-free, else the last safe q) with its gravity tauff and keep the
failed iterate out of the filter.

Use `make_safe_arm_ik()` (via xrt_compat.make_arm_ik(), which handles the cwd) or
`install_into_upstream()` to replace `teleop.robot_control.robot_arm_ik.G1_29_ArmIK`.
"""
from __future__ import annotations

import os
import sys
import time
import weakref

import numpy as np

from xr_camera_teleop.teleop import xrt_compat
from xr_camera_teleop.teleop.self_collision import MeshGate, SphereModel

xrt_compat.add_xrt_to_path()
import casadi  # noqa: E402
import pinocchio as pin  # noqa: E402
from teleop.robot_control import robot_arm_ik as _upstream  # noqa: E402

UpstreamG1_29_ArmIK = getattr(_upstream.G1_29_ArmIK, "_wilor_upstream", _upstream.G1_29_ArmIK)

# Constrained problem: upstream options (robot_arm_ik.py:160-177) + max_iter 30 -> 50 and a small initial
# barrier / bound push (the start, the current q, is normally feasible: ~2.5x fewer iterations).
IPOPT_OPTS = {
    "expand": True, "detect_simple_bounds": True, "calc_lam_p": False, "print_time": False,
    "ipopt.sb": "yes", "ipopt.print_level": 0, "ipopt.max_iter": 50, "ipopt.tol": 1e-4,
    "ipopt.acceptable_tol": 5e-4, "ipopt.acceptable_iter": 5, "ipopt.warm_start_init_point": "yes",
    "ipopt.derivative_test": "none", "ipopt.jacobian_approximation": "exact",
    "ipopt.mu_init": 1e-3, "ipopt.bound_push": 1e-6, "ipopt.warm_start_bound_push": 1e-6,
    "ipopt.warm_start_mult_bound_push": 1e-6,
}


_INSTANCES = weakref.WeakSet()


def request_home_all():
    """reset_to_home() on every live SafeG1_29_ArmIK (launcher: the IK is created inside the upstream
    script, so the wrapper's on_home_request cannot hold a direct reference)."""
    for ik in list(_INSTANCES):
        ik.reset_to_home()


def self_collision_enabled_by_env() -> bool:
    return os.environ.get("WILOR_XR_SELF_COLLISION", "1").strip().lower() not in ("0", "false", "no", "off")


class SafeG1_29_ArmIK(UpstreamG1_29_ArmIK):
    """See module docstring. Extra public API: reset_to_home(), clearance(q), mode, stats."""
    _wilor_upstream = UpstreamG1_29_ArmIK

    def __init__(self, Unit_Test=False, Visualization=False, margin=None, constraints=True, gate=True,
                 warn_period=1.0, escape=True, escape_period=0.25, escape_ratio=0.5, escape_min_gain=0.05,
                 transition_rate=2.0, stall_timeout=2.0, clock=time.monotonic):
        super().__init__(Unit_Test=Unit_Test, Visualization=Visualization)
        m = self.reduced_robot.model
        nq = m.nq
        self.sphere_model = SphereModel(margin)
        self.collision_constraints = bool(constraints)
        g = self.sphere_model.casadi_constraints(self.cdata, m.getJointId)
        self.collision_fn = casadi.Function("self_collision", [self.cq], [g])
        # tracking cost of upstream WITHOUT the smoothness term, split per arm (escape decisions)
        te = self.translational_error(self.cq, self.cTf_l, self.cTf_r)
        re = self.rotational_error(self.cq, self.cTf_l, self.cTf_r)
        JL = 50 * casadi.sumsqr(te[:3]) + casadi.sumsqr(re[:3]) + 0.02 * casadi.sumsqr(self.cq[:7])
        JR = 50 * casadi.sumsqr(te[3:]) + casadi.sumsqr(re[3:]) + 0.02 * casadi.sumsqr(self.cq[7:])
        self.cost_fn = casadi.Function("track_cost", [self.cq, self.cTf_l, self.cTf_r], [casadi.vertcat(JL, JR)])
        # upstream's own (unconstrained) problem stays as self.opti: fast path
        self.opti_free = self.opti
        self.opti_safe = self.opti_step = None
        lo, hi = m.lowerPositionLimit, m.upperPositionLimit
        if self.collision_constraints:
            o = self.opti_safe = casadi.Opti()
            self.safe_var_q = o.variable(nq)
            self.safe_q_last = o.parameter(nq)
            self.safe_tf_l = o.parameter(4, 4)
            self.safe_tf_r = o.parameter(4, 4)
            tr = casadi.sumsqr(self.translational_error(self.safe_var_q, self.safe_tf_l, self.safe_tf_r))
            rot = casadi.sumsqr(self.rotational_error(self.safe_var_q, self.safe_tf_l, self.safe_tf_r))
            o.subject_to(o.bounded(lo, self.safe_var_q, hi))
            o.subject_to(self.collision_fn(self.safe_var_q) >= 0)
            # same cost as upstream (robot_arm_ik.py:158)
            o.minimize(50 * tr + rot + 0.02 * casadi.sumsqr(self.safe_var_q)
                       + 0.1 * casadi.sumsqr(self.safe_var_q - self.safe_q_last))
            o.solver("ipopt", dict(IPOPT_OPTS))
            # collision-aware joint-space step: min |q - goal|^2, |q - q_from|_inf <= step, g(q) >= 0
            o = self.opti_step = casadi.Opti()
            self.step_q = o.variable(nq)
            self.step_goal = o.parameter(nq)
            self.step_lo = o.parameter(nq)
            self.step_hi = o.parameter(nq)
            o.subject_to(o.bounded(self.step_lo, self.step_q, self.step_hi))
            o.subject_to(self.collision_fn(self.step_q) >= 0)
            o.minimize(casadi.sumsqr(self.step_q - self.step_goal))
            o.solver("ipopt", dict(IPOPT_OPTS))
        self.gate = MeshGate() if gate else None
        self.q_home = np.zeros(nq)
        self.escape = bool(escape)
        self.escape_period, self.escape_ratio, self.escape_min_gain = escape_period, escape_ratio, escape_min_gain
        self.transition_rate, self.stall_timeout = float(transition_rate), float(stall_timeout)
        self.clock = clock
        self.warn_period = float(warn_period)
        self._last_warn = -1e9
        self.stats = {"solves": 0, "fast": 0, "constrained": 0, "solver_fail": 0, "gated": 0, "nonfinite": 0,
                      "escapes": 0, "homes": 0, "aborted": 0, "filter_fallback": 0, "last_ms": 0.0}
        self._reset_state(self.q_home)
        _INSTANCES.add(self)
        if self.collision_constraints:      # first IPOPT call of a problem is ~100 ms: pay it here
            d = m.createData()
            pin.framesForwardKinematics(m, d, self.q_home)
            T = [d.oMf[self.L_hand_id].homogeneous, d.oMf[self.R_hand_id].homogeneous]
            self._solve(self.opti_safe, self.safe_var_q, self.safe_q_last, self.safe_tf_l, self.safe_tf_r,
                        self.q_home, self.q_home, *T)
            self._step_solve(self.q_home + 0.01, self.q_home, 0.1)

    # ------------------------------------------------------------------------------------- state
    def _reset_state(self, q):
        """Fresh tracking state at posture q: warm start, smoothing filter, last safe q, mode."""
        q = np.asarray(q, float).copy()
        self.init_data = q.copy()
        self.smooth_filter._data_queue.clear()
        self.last_safe_q = q.copy()
        self.last_safe_tauff = self._tauff(q)
        self._cmd_q = q.copy()
        self.mode = "track"          # "track" | "transition" (escape to a better branch) | "home"
        self._goal = None
        self._stall_t = None
        self._t_last = None
        self._next_escape = -1e9
        self._esc_interval = self.escape_period
        self._esc_tgt = None
        self._home_request = False

    def reset_to_home(self):
        """Drive both arms back to the home posture (q=0) in joint space, rate-limited (transition_rate)
        and collision-free, then reset the IK state (warm start, filter, last safe q, escape state).
        Wrist targets are ignored until home is reached. Safe to call from another thread / a callback:
        it only sets a flag consumed by the next solve_ik()."""
        self._home_request = True

    @property
    def busy(self) -> bool:
        return self.mode != "track" or self._home_request

    def clearance(self, q) -> float:
        """Aggregated sphere-model clearance (m, conservative); < 0 = constraint violated."""
        return float(np.min(np.asarray(self.collision_fn(np.asarray(q, float)))))

    def track_cost(self, q, L, R):
        """(J_left, J_right): upstream tracking cost without the smoothness term."""
        return np.asarray(self.cost_fn(np.asarray(q, float), L, R), float).reshape(2)

    # ------------------------------------------------------------------------------------ solves
    @staticmethod
    def _solve(opti, var_q, q_last, tf_l, tf_r, seed, last, L, R):
        opti.set_initial(var_q, seed)
        opti.set_value(tf_l, L)
        opti.set_value(tf_r, R)
        opti.set_value(q_last, last)
        try:
            opti.solve()
        except Exception as e:  # noqa: BLE001  (IPOPT max_iter / infeasible / NaN)
            return None, str(e).splitlines()[-1][:80]
        q = np.asarray(opti.value(var_q), float).reshape(-1)
        return (q, None) if np.all(np.isfinite(q)) else (None, "non-finite solution")

    def _raw(self, seed, last, L, R, count=True):
        """Collision-aware IK from a seed (no filter): fast path = upstream problem, constrained problem
        only when the unconstrained optimum violates the sphere constraints. -> (q | None, err)."""
        q, err = self._solve(self.opti_free, self.var_q, self.var_q_last, self.param_tf_l, self.param_tf_r,
                             seed, last, L, R)
        if self.collision_constraints and (q is None or self.clearance(q) < 0.0):
            q, err = self._solve(self.opti_safe, self.safe_var_q, self.safe_q_last, self.safe_tf_l,
                                 self.safe_tf_r, seed, last, L, R)
            if count:
                self.stats["constrained"] += 1
        elif count:
            self.stats["fast"] += 1
        return q, err

    def _step_solve(self, q_from, goal, step):
        lo, hi = self.reduced_robot.model.lowerPositionLimit, self.reduced_robot.model.upperPositionLimit
        o = self.opti_step
        o.set_initial(self.step_q, q_from)
        o.set_value(self.step_goal, goal)
        o.set_value(self.step_lo, np.maximum(lo, q_from - step))
        o.set_value(self.step_hi, np.minimum(hi, q_from + step))
        try:
            o.solve()
        except Exception:  # noqa: BLE001
            return None
        q = np.asarray(o.value(self.step_q), float).reshape(-1)
        return q if np.all(np.isfinite(q)) else None

    def _transition_ok(self, q):
        return (q is not None and np.all(np.isfinite(q)) and self._is_safe(q)
                and (not self.collision_constraints or self.clearance(q) >= -2e-3))

    def _step_towards(self, q_from, goal, dt):
        """One rate-limited joint-space step toward goal: straight line if collision-free, else the
        collision-aware projected step. None if no safe progress is possible."""
        step = self.transition_rate * dt
        d = goal - q_from
        n = float(np.max(np.abs(d)))
        q = goal.copy() if n <= step else q_from + d * (step / n)
        if self._transition_ok(q):
            return q
        if self.opti_step is not None:
            q = self._step_solve(q_from, goal, step)
            if self._transition_ok(q) and np.max(np.abs(q - q_from)) > 1e-4:
                return q
        return None

    # ------------------------------------------------------------------------------------ helpers
    def _tauff(self, q):
        m = self.reduced_robot.model
        return pin.rnea(m, self.reduced_robot.data, np.asarray(q, float), np.zeros(m.nv), np.zeros(m.nv))

    def _warn(self, msg):
        now = time.monotonic()
        if now - self._last_warn >= self.warn_period:
            self._last_warn = now
            s = self.stats
            print(f"[SafeG1_29_ArmIK] {msg} (solves={s['solves']} constrained={s['constrained']} "
                  f"gated={s['gated']} solver_fail={s['solver_fail']} escapes={s['escapes']} homes={s['homes']})",
                  file=sys.stderr, flush=True)

    def _is_safe(self, q):
        if q is None:
            return False
        q = np.asarray(q, float)
        if q.shape != self.last_safe_q.shape or not np.all(np.isfinite(q)):
            return False
        return self.gate is None or not self.gate.collides(q)

    def _emit(self, q):
        """Gate + bookkeeping for every output."""
        if self._is_safe(q):
            self.last_safe_q = np.asarray(q, float).copy()
            self.last_safe_tauff = self._tauff(q)
            self._cmd_q = self.last_safe_q.copy()
            return self.last_safe_q.copy(), self.last_safe_tauff.copy()
        self.stats["gated"] += 1
        hit = self.gate.colliding_pairs(q) if (self.gate is not None and q is not None
                                               and np.all(np.isfinite(q))) else []
        self._warn(f"self-collision gate rejected IK output {hit[:3]}; returning last safe q")
        self.init_data = self.last_safe_q.copy()
        self._cmd_q = self.last_safe_q.copy()
        return self.last_safe_q.copy(), self.last_safe_tauff.copy()

    def _finish_transition(self, q):
        self.mode, self._goal, self._stall_t = "track", None, None
        self.init_data = q.copy()
        self.smooth_filter._data_queue.clear()     # no blending with the abandoned branch
        self.smooth_filter.add_data(q.copy())
        self._next_escape = self.clock() + self.escape_period

    def _transition_tick(self, now, dt, L, R):
        if self.mode == "transition" and L is not None:
            # keep following the operator on the good branch: warm start from the previous goal
            g, _ = self._raw(self._goal, self._goal, L, R, count=False)
            if self._transition_ok(g):
                self._goal = g
            g = self._better_branch(now, self._goal, L, R)     # the shadow itself can get stuck as well
            if g is not None:
                self._goal = g
        q = self._step_towards(self._cmd_q, self._goal, dt)
        if q is None:
            self._stall_t = now if self._stall_t is None else self._stall_t
            if now - self._stall_t > self.stall_timeout:
                self.stats["aborted"] += 1
                self._warn(f"{self.mode} blocked for {self.stall_timeout:.1f} s (no collision-free path); aborted")
                self._finish_transition(self._cmd_q.copy())
                self._next_escape = now + 4 * self.stall_timeout
            return self._emit(self._cmd_q.copy())
        self._stall_t = None
        if np.max(np.abs(q - self._goal)) < 1e-3:
            if self.mode == "home":
                self._reset_state(self._goal)
                return self._emit(self.q_home.copy())
            self._finish_transition(q)
        return self._emit(q)

    def _better_branch(self, now, q_ref, L, R):
        """q_ref (warm-started solution, or the current transition goal) stuck on a bad branch (joint-limit
        pinned / wound wrist / flipped elbow)? Compare with a solve seeded from home and return a goal that
        takes the home-seeded arm(s) where that is clearly better (cost < escape_ratio * cost and
        > escape_min_gain lower), or None. Rate-limited to one check per escape_period."""
        if not self.escape or now < self._next_escape:
            return None
        # back-off: a check that found nothing doubles the interval (up to 8x) until the targets move
        # (> 5 cm / 0.3 rad since that check): a target held inside the body costs one cold constrained
        # solve (~30-50 ms) every 2 s instead of every 0.25 s
        tgt = np.concatenate([L[:3, 3], R[:3, 3]])
        rot = np.concatenate([L[:3, :3].ravel(), R[:3, :3].ravel()])
        if self._esc_tgt is not None and (np.abs(tgt - self._esc_tgt[0]).max() > 0.05
                                          or np.abs(rot - self._esc_tgt[1]).max() > 0.3):
            self._esc_interval = self.escape_period
        self._esc_tgt = (tgt, rot)
        self._next_escape = now + self._esc_interval
        Jw = self.track_cost(q_ref, L, R)
        if Jw.sum() < self.escape_min_gain:
            return None
        cand, _ = self._raw(self.q_home, self.q_home, L, R, count=False)
        Jc = self.track_cost(cand, L, R) if cand is not None else Jw
        better = (Jc < self.escape_ratio * Jw) & (Jw - Jc > self.escape_min_gain)
        if not better.any():
            self._esc_interval = min(8 * self.escape_period, 2 * self._esc_interval)
            return None
        self._esc_interval = self.escape_period
        goal = q_ref.copy()
        if better[0]:
            goal[:7] = cand[:7]
        if better[1]:
            goal[7:] = cand[7:]
        if not self._transition_ok(goal):
            goal = cand
            if not self._transition_ok(goal) or Jc.sum() > self.escape_ratio * Jw.sum():
                return None
        self.stats["escapes"] += 1
        self._warn(f"IK stuck on a worse branch (cost L/R {Jw.round(3)} vs {Jc.round(3)} from home); "
                   f"moving to the better one at {self.transition_rate} rad/s")
        return goal

    # ------------------------------------------------------------------------------------ main API
    def solve_ik(self, left_wrist, right_wrist, current_lr_arm_motor_q=None, current_lr_arm_motor_dq=None):
        self.stats["solves"] += 1
        now = self.clock()
        dt = 1.0 / 30 if self._t_last is None else float(np.clip(now - self._t_last, 1e-3, 0.1))
        self._t_last = now
        cur = None
        if current_lr_arm_motor_q is not None:
            c = np.asarray(current_lr_arm_motor_q, float).copy()
            if c.shape == self.last_safe_q.shape and np.all(np.isfinite(c)):
                cur = c
        L = np.asarray(left_wrist, float)
        R = np.asarray(right_wrist, float)
        targets_ok = L.shape == (4, 4) and R.shape == (4, 4) and np.all(np.isfinite(L)) and np.all(np.isfinite(R))
        if not targets_ok:
            self.stats["nonfinite"] += 1
            self._warn("non-finite wrist target; holding")
        t0 = time.perf_counter()
        if self._home_request:
            self._home_request = False
            self.stats["homes"] += 1
            self.mode, self._goal, self._stall_t = "home", self.q_home.copy(), None
            if cur is not None and self._is_safe(cur):
                self._cmd_q = cur.copy()
        if self.mode in ("home", "transition"):
            out = self._transition_tick(now, dt, L if targets_ok else None, R)
            self.stats["last_ms"] = (time.perf_counter() - t0) * 1e3
            return out
        # ---------------- normal tracking (upstream semantics: warm start from the current motor q)
        if cur is not None:
            self.init_data = cur
        if not np.all(np.isfinite(self.init_data)):
            self.init_data = self.last_safe_q.copy()
        q = None
        if targets_ok:
            q_raw, err = self._raw(self.init_data, self.init_data, L, R)
            if err is not None:
                self.stats["solver_fail"] += 1
                self._warn(f"IK did not converge ({err}); holding current q")
            if q_raw is not None:
                self.smooth_filter.add_data(q_raw)
                q = np.asarray(self.smooth_filter.filtered_data, float).copy()
                if not self._is_safe(q) and self._is_safe(q_raw):
                    # upstream's 4-tap moving average of collision-free solutions can cut through the body
                    # when the solution moves fast: take the first safe point from the average toward the raw
                    self.stats["filter_fallback"] += 1
                    for t in (0.25, 0.5, 0.75, 1.0):
                        qt = q + t * (q_raw - q)
                        if self._is_safe(qt):
                            q = qt
                            break
                goal = self._better_branch(now, q_raw, L, R)
                if goal is not None:
                    self.mode, self._goal, self._stall_t = "transition", goal, None
        self.stats["last_ms"] = (time.perf_counter() - t0) * 1e3
        if q is None:          # upstream semantics on failure: hold the current motor q
            q = cur if cur is not None else self.init_data.copy()
        out = self._emit(q)
        self.init_data = out[0].copy()
        return out


def make_safe_arm_ik(**kw):
    return SafeG1_29_ArmIK(**kw)


def install_into_upstream(**kw):
    """Replace G1_29_ArmIK in xr_teleoperate's robot_arm_ik module, so that a LATER
    `from teleop.robot_control.robot_arm_ik import G1_29_ArmIK` (teleop_hand_and_arm.py) yields the safe class.
    kw (margin=, constraints=, gate=) are baked into the constructor."""
    if kw:
        cls = type("SafeG1_29_ArmIK_cfg", (SafeG1_29_ArmIK,),
                   {"__init__": lambda self, *a, **k: SafeG1_29_ArmIK.__init__(self, *a, **{**kw, **k})})
    else:
        cls = SafeG1_29_ArmIK
    _upstream.G1_29_ArmIK = cls
    return cls
