"""Self-collision protection: sphere model (IK constraints), mesh gate, SafeG1_29_ArmIK vs upstream IK."""
import time

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("casadi")
pytest.importorskip("hppfcl")

from xr_camera_teleop.teleop import g1_collision_spheres as GS  # noqa: E402
from xr_camera_teleop.teleop import xrt_compat  # noqa: E402
from xr_camera_teleop.teleop.clutch import neutral_from_ik_fk  # noqa: E402
from xr_camera_teleop.teleop.filters import rot_exp  # noqa: E402
from xr_camera_teleop.teleop.self_collision import MeshGate, SphereModel, build_reduced_models, is_hand_wrist  # noqa: E402

# right arm joint indices in the 14-vector: 7 pitch, 8 roll, 9 yaw, 10 elbow, 11-13 wrist
R_ROLL, R_YAW = 8, 9


@pytest.fixture(scope="module")
def gate():
    return MeshGate()


@pytest.fixture(scope="module")
def spheres():
    return SphereModel()


class TickClock:
    """Virtual clock: +1/30 s per read (the IK reads it once per solve_ik = 30 Hz control loop)."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 1 / 30
        return self.t


@pytest.fixture(scope="module")
def safe_ik():
    ik = xrt_compat.make_arm_ik(self_collision=True)
    ik.clock = TickClock()
    return ik


@pytest.fixture(scope="module")
def noescape_ik():
    ik = xrt_compat.make_arm_ik(self_collision=True, escape=False)
    ik.clock = TickClock()
    return ik


@pytest.fixture(scope="module")
def gate_all():
    """Mesh check that ALSO counts hand-vs-hand contact (to measure it; the real gate allows it)."""
    return MeshGate(allow_hand_contact=False)


@pytest.fixture(scope="module")
def up_ik():
    return xrt_compat.make_arm_ik(self_collision=False)


@pytest.fixture(scope="module")
def neutral(up_ik):
    return neutral_from_ik_fk(up_ik)


def pose(T, p):
    T = T.copy()
    T[:3, 3] = p
    return T


def reset(ik):
    """Fresh tracking state (as after construction): q=0, empty smoothing filter."""
    ik.init_data = np.zeros(14)
    ik.smooth_filter._data_queue.clear()
    if hasattr(ik, "_reset_state"):
        ik._reset_state(np.zeros(14))


def track(ik, L, R, n=40, q=None):
    """Feed the same target n ticks, robot follows the IK output exactly (kinematic). Returns all q."""
    reset(ik)
    q = np.zeros(14) if q is None else q
    out = []
    for _ in range(n):
        q, tau = ik.solve_ik(L, R, q, np.zeros(14))
        assert q.shape == (14,) and tau.shape == (14,)
        out.append(np.asarray(q, float).copy())
    return out


def ee_err(ik, q, L, R):
    Lq, Rq = neutral_from_ik_fk(ik, q)
    return np.linalg.norm(Lq[:3, 3] - L[:3, 3]), np.linalg.norm(Rq[:3, 3] - R[:3, 3])


# ---------------------------------------------------------------------------------------- geometry
def test_reduced_model_matches_ik_model(up_ik):
    import pinocchio as pin

    red, _ = build_reduced_models()
    m = up_ik.reduced_robot.model
    assert list(red.names) == list(m.names)
    q = np.random.default_rng(1).uniform(m.lowerPositionLimit, m.upperPositionLimit)
    d1, d2 = red.createData(), m.createData()
    pin.forwardKinematics(red, d1, q)
    pin.forwardKinematics(m, d2, q)
    for j in range(1, m.njoints):
        np.testing.assert_allclose(d1.oMi[j].homogeneous, d2.oMi[j].homogeneous, atol=1e-9)


def test_primitives_cover_the_meshes():
    """Every collision-mesh vertex lies inside the fitted spheres (arm) / capsules (body)."""
    import pinocchio as pin

    from xr_camera_teleop.tools.fit_self_collision import ARM_GROUPS, BODY, geom_vertices

    red, geom = build_reduced_models()
    by_name = {g.name: g for g in geom.geometryObjects}
    for s in ("left", "right"):
        for grp, jn, links, _ in ARM_GROUPS:
            sph = [(np.array(c), r) for side, j, g, c, r in GS.ARM_SPHERES if side == s and g == grp]
            X = np.concatenate([geom_vertices(by_name[f"{s}_{l}_0"]) for l in links])
            d = np.min([np.linalg.norm(X - c, axis=1) - r for c, r in sph], axis=0)
            assert d.max() < 1e-6, (s, grp, d.max())
    caps = [(np.array(a), np.array(b), r) for _, a, b, r in GS.BODY_CAPSULES]

    def dcap(X, a, b, r):
        ab = b - a
        t = np.clip((X - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        return np.linalg.norm(X - a - t[:, None] * ab, axis=1) - r

    links = sorted({l for _, ls, _, _ in BODY for l in ls})
    for l in links:
        X = geom_vertices(by_name[f"{l}_0"])
        d = np.min([dcap(X, *c) for c in caps], axis=0)
        assert d.max() < 2e-3, (l, d.max())
    assert pin is not None


def test_home_pose_collision_free(gate, spheres, safe_ik):
    q0 = np.zeros(14)
    assert gate.colliding_pairs(q0) == []
    assert np.min(spheres.clearances(q0)) >= 0.0
    assert safe_ik.clearance(q0) >= 0.0          # aggregated IK constraints feasible at home


def test_typical_poses_collision_free(gate, spheres):
    typical = {
        "arms_forward": [-0.8, 0.2, 0, 0.6, 0, 0, 0, -0.8, -0.2, 0, 0.6, 0, 0, 0],
        "arms_raised_side": [0, 1.0, 0, 0.3, 0, 0, 0, 0, -1.0, 0, 0.3, 0, 0, 0],
        "elbows_bent_up": [-0.3, 0.2, 0, -0.6, 0, 0, 0, -0.3, -0.2, 0, -0.6, 0, 0, 0],
    }
    for name, q in typical.items():
        assert gate.colliding_pairs(q) == [], name
        assert np.min(spheres.clearances(q)) >= 0.0, (name, spheres.violated(q)[:3])


@pytest.mark.parametrize("j,val", [(R_ROLL, 0.6), (R_ROLL, 1.0), (R_YAW, 1.2)])
def test_right_arm_in_torso_detected_by_both_models(gate, spheres, j, val):
    q = np.zeros(14)
    q[j] = val
    hits = gate.colliding_pairs(q)
    assert any(b == "torso_link_0" and a.startswith("right_") for a, b in hits), hits
    assert spheres.collides(q)
    assert gate.collides(q)


def test_gate_rejects_nonfinite(gate):
    q = np.zeros(14)
    q[3] = np.nan
    assert gate.collides(q)
    assert gate.collides(np.zeros(13))


# ---------------------------------------------------------------------------------------------- IK
def scenarios(neutral):
    L0, R0 = neutral
    return {
        "right_in_torso": (L0, pose(R0, [0.0, -0.02, 0.15])),
        "left_in_torso": (pose(L0, [0.0, 0.02, 0.15]), R0),
        "both_in_torso": (pose(L0, [0.02, 0.0, 0.2]), pose(R0, [0.02, 0.0, 0.2])),
        "hands_into_each_other": (pose(L0, [0.30, -0.05, 0.12]), pose(R0, [0.30, 0.05, 0.12])),
        "hands_crossed": (pose(L0, [0.25, -0.12, 0.12]), pose(R0, [0.25, 0.12, 0.12])),
        "forearms_cross": (pose(L0, [0.30, -0.30, 0.15]), pose(R0, [0.30, 0.30, 0.15])),
        "right_behind_back": (L0, pose(R0, [-0.15, -0.05, 0.12])),
        "right_into_head": (L0, pose(R0, [0.02, -0.01, 0.45])),
    }


def test_ik_targets_inside_body_stay_collision_free(safe_ik, up_ik, gate, neutral):
    up_collided = []
    for name, (L, R) in scenarios(neutral).items():
        g0 = safe_ik.stats["gated"]
        qs = track(safe_ik, L, R)
        for k, q in enumerate(qs):
            assert gate.colliding_pairs(q) == [], (name, k, gate.colliding_pairs(q))
        # layer 1 alone does the job: the gate never had to intervene
        assert safe_ik.stats["gated"] == g0, name
        qu = track(up_ik, L, R)
        if any(gate.collides(q) for q in qu):
            up_collided.append(name)
    print("upstream IK collides in:", up_collided)
    # the scenarios are meaningful: the unprotected IK goes through the body in most of them
    assert {"right_in_torso", "left_in_torso", "hands_into_each_other", "forearms_cross"} <= set(up_collided)
    assert len(up_collided) >= 4


def test_arm_stops_at_the_surface(safe_ik, spheres, neutral):
    L0, R0 = neutral
    R = pose(R0, [0.0, -0.02, 0.15])
    q = track(safe_ik, L0, R, n=60)[-1]
    # active contact: some right-arm pair sits on its threshold (stopped AT the surface, not short of it)
    cl = spheres.clearances(q)
    assert cl.min() > -2e-3 and cl.min() < 0.01
    # and the hand got as close as it could: the left arm stays where it was asked
    el, er = ee_err(safe_ik, q, L0, R)
    assert el < 0.01 and 0.03 < er < 0.3


def test_free_space_targets_tracked_like_upstream(safe_ik, up_ik, neutral):
    L0, R0 = neutral
    offsets = [([0.10, 0.0, 0.10], [0.10, 0.0, 0.10]), ([0.0, 0.10, 0.15], [0.0, -0.10, 0.15]),
               ([0.15, 0.05, -0.05], [0.12, -0.05, 0.0]), ([-0.05, 0.08, 0.0], [0.05, -0.10, 0.20]),
               ([0.05, 0.0, 0.25], [0.0, 0.0, 0.0])]
    reached = 0
    for dl, dr in offsets:
        L = pose(L0, L0[:3, 3] + dl)
        R = pose(R0, R0[:3, 3] + dr)
        qs, qu = track(safe_ik, L, R)[-1], track(up_ik, L, R)[-1]
        es, eu = ee_err(safe_ik, qs, L, R), ee_err(up_ik, qu, L, R)
        print(f"  target dl={dl} dr={dr}: err safe {np.round(es, 4)} upstream {np.round(eu, 4)}")
        for a, b in zip(es, eu):
            assert a < b + 2e-3, (dl, dr, es, eu)
            if b < 0.01:                      # reachable for upstream -> reached by the safe IK too
                assert a < 0.01, (dl, dr, es, eu)
                reached += 1
    assert reached >= 8


def test_trajectory_through_torso_no_collision_no_jumps(safe_ik, gate, neutral):
    L0, R0 = neutral
    reset(safe_ik)
    a, b = R0[:3, 3], np.array([-0.10, 0.08, 0.15])          # free space -> through torso -> left side
    path = [a + (b - a) * s for s in np.linspace(0, 1, 120)] + [b + (a - b) * s for s in np.linspace(0, 1, 120)]
    q = np.zeros(14)
    g0, jumps = safe_ik.stats["gated"], []
    for p in path:
        qn, _ = safe_ik.solve_ik(L0, pose(R0, p), q, np.zeros(14))
        assert gate.colliding_pairs(qn) == [], p
        jumps.append(np.abs(qn - q).max())
        q = qn
    assert safe_ik.stats["gated"] == g0
    print(f"max per-tick joint step {max(jumps):.3f} rad")
    assert max(jumps) < 0.15
    el, er = ee_err(safe_ik, q, L0, R0)          # back in free space: tracking again
    assert er < 0.01


def test_gate_returns_last_safe_q(neutral, gate):
    ik = xrt_compat.make_arm_ik(self_collision=True, constraints=False, escape=False)    # layer 2 only
    L0, R0 = neutral
    q = np.zeros(14)
    q, _ = ik.solve_ik(L0, pose(R0, R0[:3, 3] + [0.05, 0, 0.05]), q)        # a safe pose first
    safe = q.copy()
    assert not gate.collides(safe)
    for _ in range(30):
        qn, tau = ik.solve_ik(L0, pose(R0, [0.0, -0.02, 0.15]), q)
        assert not gate.collides(qn)
        q = qn
    assert ik.stats["gated"] > 0
    np.testing.assert_allclose(qn, ik.last_safe_q)
    np.testing.assert_allclose(tau, ik.last_safe_tauff)
    # direct: a colliding "solution" is never returned
    bad = np.zeros(14)
    bad[R_ROLL] = 1.0
    ik.last_safe_q = safe.copy()
    ik.last_safe_tauff = ik._tauff(safe)
    ik.escape = False
    orig = ik._raw
    ik._raw = lambda *a, **k: (bad.copy(), None)
    try:
        qn, _ = ik.solve_ik(L0, R0, safe)
        qn, _ = ik.solve_ik(L0, R0, safe)       # filter: the bad q dominates
    finally:
        ik._raw = orig
    np.testing.assert_allclose(qn, safe)
    # NaN target -> holds (current q is safe) ; NaN current q -> last safe
    Tn = R0.copy()
    Tn[0, 3] = np.nan
    qn, tau = ik.solve_ik(L0, Tn, safe)
    np.testing.assert_allclose(qn, safe)
    assert np.all(np.isfinite(tau))
    qn, _ = ik.solve_ik(L0, Tn, np.full(14, np.nan))
    np.testing.assert_allclose(qn, ik.last_safe_q)


def test_timing_benchmark(safe_ik, up_ik, neutral):
    L0, R0 = neutral
    res = {}
    for tag, R in (("free", pose(R0, R0[:3, 3] + [0.08, 0.0, 0.08])), ("contact", pose(R0, [0.0, -0.02, 0.15]))):
        for name, ik in (("safe", safe_ik), ("upstream", up_ik)):
            reset(ik)
            q = np.zeros(14)
            ts = []
            for k in range(60):
                s = min(1.0, k / 30)
                T = pose(R0, R0[:3, 3] + (R[:3, 3] - R0[:3, 3]) * s)
                t0 = time.perf_counter()
                q, _ = ik.solve_ik(L0, T, q)
                ts.append((time.perf_counter() - t0) * 1e3)
            res[(tag, name)] = (np.median(ts), np.percentile(ts, 95), np.max(ts))
    t0 = time.perf_counter()
    for _ in range(200):
        safe_ik.gate.collides(np.zeros(14))
    gate_ms = (time.perf_counter() - t0) * 1e3 / 200
    print()
    for (tag, name), (med, p95, mx) in res.items():
        print(f"  IK {tag:8s} {name:9s} median {med:6.2f} ms  p95 {p95:6.2f} ms  max {mx:6.2f} ms")
    print(f"  mesh gate {gate_ms:.3f} ms/check")
    assert res[("free", "safe")][0] < 20 and res[("contact", "safe")][0] < 40


# --------------------------------------------------------------------------------------- end to end
def test_end_to_end_synthetic_stream_push_right_hand_into_body(safe_ik, gate):
    """Synthetic hand stream -> WiLoRTeleWrapper (ego clutch, filters) -> SafeG1_29_ArmIK, operator pushes
    the right hand into their chest and across to the left; the robot never self-collides."""
    from xr_camera_teleop.teleop import synthetic_stream as syn
    from xr_camera_teleop.teleop.wilor_tv_wrapper import WiLoRTeleWrapper

    class Client:
        msg = None

        def latest(self, max_age=0.3):
            return self.msg

    clk = [100.0]
    cl = Client()
    reset(safe_ik)
    w = WiLoRTeleWrapper(client=cl, clock=lambda: clk[0], k=1.0, neutral=neutral_from_ik_fk(safe_ik),
                         auto_calibrate=False)
    rest = {s: (syn.REST[s].copy(), syn.rest_rotation(s)) for s in ("left", "right")}
    q = np.zeros(14)
    worst_target_x = 1.0
    g0 = safe_ik.stats["gated"]
    n = 0
    for k in range(240):
        clk[0] += 1 / 30
        poses = dict(rest)
        if k > 20:          # right hand: back toward the operator's body and across to the left
            s = min(1.0, (k - 20) / 120)
            poses["right"] = (rest["right"][0] + s * np.array([-0.32, 0.22, 0.05]), rest["right"][1])
        cl.msg = syn.make_message(k, clk[0], {h: syn.hand_message_entry(h, p, R) for h, (p, R) in poses.items()})
        if k == 2:
            w.calibrate()
        td = w.get_tele_data()
        worst_target_x = min(worst_target_x, td.right_wrist_pose[0, 3])
        q, _ = safe_ik.solve_ik(td.left_wrist_pose, td.right_wrist_pose, q, np.zeros(14))
        assert gate.colliding_pairs(q) == [], k
        n += 1
    w.close()
    assert w.clutch["right"].calibrated
    assert worst_target_x < 0.0               # the commanded right wrist really was inside the torso
    assert safe_ik.stats["gated"] == g0


# ------------------------------------------------------------------------------ hands may touch
def test_gate_and_constraints_allow_hand_hand_contact(gate, spheres):
    assert not any(is_hand_wrist(a) and is_hand_wrist(b) for a, b in gate.pairs)
    assert any("elbow" in a and is_hand_wrist(b) for a, b in gate.pairs)        # forearm vs other hand kept
    hw = set(GS.HAND_WRIST_GROUPS)
    for kind, i, j, _ in spheres.pairs:
        if kind == "arm":
            assert not (spheres.spheres[i][2] in hw and spheres.spheres[j][2] in hw)


def test_hands_together_reach_contact(safe_ik, gate, gate_all, neutral):
    """Clap / wrist on wrist: the robot hands get into contact (the old constraint kept them ~3 cm apart),
    and nothing else collides."""
    L0, R0 = neutral
    for p_l, p_r in (([0.30, 0.0, 0.12], [0.30, 0.0, 0.12]), ([0.30, -0.05, 0.12], [0.30, 0.05, 0.12])):
        L, R = pose(L0, p_l), pose(R0, p_r)
        g0 = safe_ik.stats["gated"]
        qs = track(safe_ik, L, R, n=60)
        assert all(not gate.collides(q) for q in qs)
        assert safe_ik.stats["gated"] == g0
        d, pair = gate_all.min_distance(qs[-1], lambda a, b: is_hand_wrist(a) and is_hand_wrist(b)
                                        and a.startswith("left") and b.startswith("right"))
        print(f"  hands target {p_l}/{p_r}: hand-hand mesh distance {d * 100:.1f} cm {pair}")
        assert d < 0.005


def test_forearm_through_forearm_still_blocked(safe_ik, up_ik, gate, neutral):
    L0, R0 = neutral
    L, R = pose(L0, [0.30, -0.30, 0.15]), pose(R0, [0.30, 0.30, 0.15])

    def arm_arm(q):
        return [(a, b) for a, b in gate.colliding_pairs(q) if a.startswith("left") and b.startswith("right")]

    assert any(arm_arm(q) for q in track(up_ik, L, R))
    assert not any(arm_arm(q) or gate.collides(q) for q in track(safe_ik, L, R))


def test_torso_stop_distance(safe_ik, gate, neutral):
    """Hand pushed into the chest stops close to it (sphere/capsule conservativeness + 5 mm margin)."""
    L0, R0 = neutral
    q = track(safe_ik, L0, pose(R0, [0.0, -0.02, 0.15]), n=60)[-1]
    d, pair = gate.min_distance(q, lambda a, b: a.startswith("right") and "shoulder" not in a
                                and b in ("torso_link_0", "logo_link_0"))
    print(f"  right hand/forearm to torso: {d * 100:.1f} cm {pair}")
    assert 0.0 < d < 0.03


# ------------------------------------------------------------------------------ stuck branches
def _seq_via(ways, n=30, hold=90):
    from scipy.spatial.transform import Rotation, Slerp

    out = []
    for a, b in zip(ways[:-1], ways[1:]):
        for s in np.linspace(0, 1, n):
            pair = []
            for A, B in zip(a, b):
                T = np.eye(4)
                T[:3, :3] = Slerp([0, 1], Rotation.from_matrix([A[:3, :3], B[:3, :3]]))(s).as_matrix()
                T[:3, 3] = (1 - s) * A[:3, 3] + s * B[:3, 3]
                pair.append(T)
            out.append(tuple(pair))
    return out + [ways[-1]] * hold


def _excursions(neutral, n_trials, seed=0):
    """Operator returns to a normal working pose after 3 random unusual hand poses (positions +-30 cm,
    wrist rotations ~N(0, 1.2 rad) per axis): the cases that left the arms twisted."""
    L0, R0 = neutral
    rng = np.random.default_rng(seed)

    def rnd(T0):
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] + rng.uniform([-0.3, -0.3, -0.3], [0.25, 0.3, 0.4])
        T[:3, :3] = rot_exp(rng.normal(0, 1.2, 3)) @ T0[:3, :3]
        return T

    work = pose(L0, L0[:3, 3] + [0.1, 0, 0.15]), pose(R0, R0[:3, 3] + [0.1, 0, 0.15])
    for _ in range(n_trials):
        yield work, _seq_via([work] + [(rnd(L0), rnd(R0)) for _ in range(3)] + [work])


def _run_seq(ik, seq):
    reset(ik)
    q, steps = np.zeros(14), []
    for L, R in seq:
        qn, _ = ik.solve_ik(L, R, q, np.zeros(14))
        steps.append(np.abs(qn - q).max())
        q = qn
    return q, steps


def test_escape_from_stuck_branch(safe_ik, noescape_ik, neutral, gate):
    """Root cause repro: warm-started IK (current q seed + |q - q_last| term + joint limits) stays on a
    joint-limit-pinned / wound branch after unusual hand poses, even when the hands are back at a normal
    pose. Without escape most trials stay stuck (cm..dm error, tens of degrees); with escape all recover."""
    stuck = {"escape": 0, "no_escape": 0}
    for (L, R), seq in _excursions(neutral, 8):
        ref = track(safe_ik, L, R, n=60)[-1]                    # fresh solve from home
        J_ref = safe_ik.track_cost(ref, L, R).sum()
        for tag, ik in (("escape", safe_ik), ("no_escape", noescape_ik)):
            g0 = ik.stats["gated"]
            q, steps = _run_seq(ik, seq)
            J = safe_ik.track_cost(q, L, R).sum()
            if J > J_ref + 0.05:
                stuck[tag] += 1
            if tag == "escape":
                assert ik.stats["gated"] == g0
                tail = steps[-90:]                              # hands back at the working pose
                assert max(tail) < 2.0 / 30 + 1e-3 or max(tail) < 0.1, max(tail)
                assert not gate.collides(q)
    print(f"  stuck after excursions: {stuck}")
    assert stuck["escape"] == 0
    assert stuck["no_escape"] >= 3


# ------------------------------------------------------------------------------ reset to home
def _twisted(safe_ik, neutral):
    L0, R0 = neutral
    T = pose(R0, [0.15, 0.25, 0.25])
    T[:3, :3] = rot_exp(np.array([2.5, 0.5, 0])) @ R0[:3, :3]
    return track(safe_ik, pose(L0, [0.05, 0.3, 0.3]), T, n=60)[-1]


def test_reset_to_home_joint_space(safe_ik, gate, neutral):
    L0, R0 = neutral
    q = _twisted(safe_ik, neutral)
    assert np.abs(q).max() > 1.0
    safe_ik.reset_to_home()
    far_L, far_R = pose(L0, [0.4, 0.3, 0.3]), pose(R0, [0.4, -0.3, 0.0])     # targets ignored while homing
    n, prev = 0, q.copy()
    while True:
        qn, tau = safe_ik.solve_ik(far_L, far_R, prev, np.zeros(14))
        assert not gate.collides(qn)
        assert np.abs(qn - prev).max() <= safe_ik.transition_rate / 30 + 1e-6      # rate limited
        assert np.all(np.isfinite(tau))
        prev, n = qn, n + 1
        if safe_ik.mode == "track":
            break
        assert n < 200
    np.testing.assert_allclose(prev, 0.0, atol=1e-9)
    assert n >= int(np.abs(q).max() / (safe_ik.transition_rate / 30)) - 1
    np.testing.assert_allclose(safe_ik.init_data, 0.0)
    np.testing.assert_allclose(safe_ik.last_safe_q, 0.0)
    assert len(safe_ik.smooth_filter._data_queue) == 0
    # tracking resumes normally afterwards
    q, _ = safe_ik.solve_ik(L0, R0, prev)
    assert np.abs(q).max() < 1e-2 and safe_ik.mode == "track"


def test_wrapper_reset_and_swap_drive_home(safe_ik, neutral):
    """live_ui Reset button -> wrapper.reset() -> on_home_request -> SafeG1_29_ArmIK.reset_to_home."""
    from xr_camera_teleop.teleop import synthetic_stream as syn
    from xr_camera_teleop.teleop.wilor_tv_wrapper import WiLoRTeleWrapper

    class Client:
        msg = None

        def latest(self, max_age=0.3):
            return self.msg

    clk = [100.0]
    w = WiLoRTeleWrapper(client=Client(), clock=lambda: clk[0], k=1.0, neutral=neutral)
    xrt_compat.connect_ik(safe_ik, w)
    q = _twisted(safe_ik, neutral)
    w.reset()
    assert safe_ik.busy
    td = w.get_tele_data()
    for _ in range(200):
        q, _ = safe_ik.solve_ik(td.left_wrist_pose, td.right_wrist_pose, q)
        if not safe_ik.busy:
            break
    np.testing.assert_allclose(q, 0.0, atol=1e-9)
    # request_reset() (signal-safe path used by the launcher) and the L/R swap latch call it as well
    calls = []
    w.on_home_request = lambda: calls.append(1)
    w.request_reset()
    w.get_tele_data()
    assert calls == [1]
    w._heading_known = True
    rest = {s: (syn.REST[s].copy(), syn.rest_rotation(s)) for s in ("left", "right")}
    swapped = {"left": rest["right"], "right": rest["left"]}
    for k in range(12):
        clk[0] += 1 / 30
        w.client.msg = syn.make_message(1000 + k, clk[0], {h: syn.hand_message_entry(h, p, R)
                                                           for h, (p, R) in swapped.items()})
        w.get_tele_data()
    assert w.swapped and calls == [1, 1]
    w.close()
