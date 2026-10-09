"""G1 (29 dof + Dex3) arm self-collision geometry, shared by the safe IK and the post-solve gate.

Two independent models of the same robot, both on the 14-dof arm-only reduced model that
xr_teleoperate's G1_29_ArmIK optimises (legs, waist and finger joints locked at 0, pelvis = origin):

* SphereModel (layer 1, used INSIDE the IK as smooth IPOPT constraints): every arm link is covered by
  spheres attached to its parent joint, the fixed body (torso, head, pelvis, thighs) by capsules in the
  pelvis frame. Both are FITTED to the URDF collision meshes by `xr_camera_teleop/tools/fit_self_collision.py`,
  which writes the numbers to `g1_collision_spheres.py` (every mesh vertex is inside the primitives).
  Constraint per checked pair:  dist^2 - (r_i + r_j + margin)^2 >= 0  (margin 5 mm vs body, 1 cm arm-arm)  (point-point or point-segment
  squared distance: C1, no sqrt). Pairs that are already close at q=0 (upper arm next to the torso:
  real mesh gap 8.6 mm) get their threshold lowered to (home distance - HOME_SLACK) so that the home
  pose stays feasible; the generated file lists them.
* MeshGate (layer 2, run on every IK output): hpp-fcl collision on the real URDF collision meshes
  (pinocchio GeometryModel), arm links vs body and left arm vs right arm. ~0.1 ms per check.
Both layers ALLOW left-hand/wrist vs right-hand/wrist contact (operator request: clapping, wrist on wrist).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
G1_DIR = REPO / "third_party" / "xr_teleoperate" / "assets" / "g1"
G1_URDF = G1_DIR / "g1_body29_hand14.urdf"
SIDES = ("left", "right")
ARM_PREFIXES = tuple(f"{s}_{p}" for s in SIDES for p in ("shoulder", "elbow", "wrist"))

# Mesh geometries of the fixed body checked against the arms (pinocchio geometry names = <link>_0).
BODY_MESHES = ["torso_link_0", "head_link_0", "logo_link_0", "pelvis_contour_link_0"] + [
    f"{s}_{l}_link_0" for s in SIDES for l in ("hip_pitch", "hip_roll", "hip_yaw", "knee")]
# Arm geometries never checked against the body: the shoulder pitch/roll cylinders are attached to
# the torso (always adjacent). They ARE checked against the other arm.
ARM_NOT_VS_BODY = ("shoulder_pitch_link", "shoulder_roll_link")
# Hand/wrist meshes (wrist roll/pitch/yaw links, palm, fingers, thumb): left-vs-right contact between two
# of these is ALLOWED (clapping, wrist on wrist) in both layers. Hand vs body and hand vs the other
# arm's upper arm / elbow (forearm) stay checked.
HAND_WRIST_KEYS = ("wrist", "hand")


def is_hand_wrist(name: str) -> bool:
    return any(k in name for k in HAND_WRIST_KEYS)

SOFTMIN_BETA = 500.0      # 1/m, sharpness of the per-sphere soft-min aggregation (IK constraints)

_MODELS = None


def build_reduced_models():
    """(reduced_model, collision GeometryModel) on the 14 arm joints, same lock set/reference (q=0)
    as G1_29_ArmIK. Cached per process."""
    global _MODELS
    if _MODELS is None:
        import pinocchio as pin

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full, cgeom, _ = pin.buildModelsFromUrdf(str(G1_URDF), str(G1_DIR))
        lock = [full.getJointId(n) for n in full.names[1:] if not n.startswith(ARM_PREFIXES)]
        red, geom = pin.buildReducedModel(full, cgeom, lock, np.zeros(full.nq))
        _MODELS = (red, geom)
    return _MODELS


def arm_mesh_names(geom, side):
    return [g.name for g in geom.geometryObjects
            if g.name.startswith(side + "_") and any(k in g.name for k in ("shoulder", "elbow", "wrist", "hand"))]


# ----------------------------------------------------------------------------------- layer 2: meshes
class MeshGate:
    """Real-mesh self-collision check (hpp-fcl via pinocchio). `security_margin` (m) inflates contact.
    allow_hand_contact=False also checks left-hand/wrist vs right-hand/wrist (diagnostics/tests only)."""

    def __init__(self, security_margin: float = 0.0, allow_hand_contact: bool = True):
        import pinocchio as pin

        self.pin = pin
        red, geom0 = build_reduced_models()
        self.model = red
        self.geom = geom0.copy()
        g = self.geom
        self.pairs = []
        for s in SIDES:
            for a in arm_mesh_names(g, s):
                if any(k in a for k in ARM_NOT_VS_BODY):
                    continue
                for b in BODY_MESHES:
                    self.pairs.append((a, b))
        for a in arm_mesh_names(g, "left"):
            for b in arm_mesh_names(g, "right"):
                if not (allow_hand_contact and is_hand_wrist(a) and is_hand_wrist(b)):
                    self.pairs.append((a, b))
        for a, b in self.pairs:
            g.addCollisionPair(pin.CollisionPair(g.getGeometryId(a), g.getGeometryId(b)))
        self.data = red.createData()
        self.gdata = pin.GeometryData(g)
        for req in self.gdata.collisionRequests:
            req.security_margin = float(security_margin)

    def colliding_pairs(self, q):
        """List of (geom_a, geom_b) in contact at q (empty = collision-free)."""
        q = np.asarray(q, float)
        self.pin.computeCollisions(self.model, self.data, self.geom, self.gdata, q, False)
        return [self.pairs[k] for k, r in enumerate(self.gdata.collisionResults) if r.isCollision()]

    def collides(self, q) -> bool:
        q = np.asarray(q, float)
        if q.shape != (self.model.nq,) or not np.all(np.isfinite(q)):
            return True
        return bool(self.pin.computeCollisions(self.model, self.data, self.geom, self.gdata, q, True))

    def min_distance(self, q, select=lambda a, b: True):
        """(distance [m], pair) minimum over the checked pairs accepted by select(a, b). Slow (~1 ms/pair):
        diagnostics and tests only."""
        pin = self.pin
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata, np.asarray(q, float))
        best = (np.inf, None)
        for k, (a, b) in enumerate(self.pairs):
            if select(a, b):
                d = pin.computeDistance(self.geom, self.gdata, k).min_distance
                if d < best[0]:
                    best = (float(d), (a, b))
        return best


# ---------------------------------------------------------------------------------- layer 1: spheres
def _seg_dist2_np(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0)
    d = p - a - t * ab
    return float(np.dot(d, d))


class SphereModel:
    """Sphere/capsule self-collision model from g1_collision_spheres.py (see module docstring)."""

    def __init__(self, margin: float | None = None):
        from xr_camera_teleop.teleop import g1_collision_spheres as S

        self.S = S
        # margin: None = generated defaults (S.MARGIN = {"body": .., "arm": ..}), float or dict overrides
        m = dict(S.MARGIN)
        if margin is not None:
            m.update(margin if isinstance(margin, dict) else {"body": float(margin), "arm": float(margin)})
        self.margin = m
        self.spheres = S.ARM_SPHERES          # (side, joint_name, group, center[3], radius)
        self.capsules = S.BODY_CAPSULES       # (name, a[3], b[3], radius)  pelvis frame
        # pairs: ("body", sphere_i, capsule_j, threshold) / ("arm", sphere_i, sphere_j, threshold)
        self.pairs = [(kind, i, j, thr + m[kind] - S.MARGIN[kind]) for kind, i, j, thr in S.PAIRS]
        red, _ = build_reduced_models()
        self.model = red
        self.data = red.createData()
        self.jid = [red.getJointId(s[1]) for s in self.spheres]

    def centers(self, q):
        import pinocchio as pin

        pin.forwardKinematics(self.model, self.data, np.asarray(q, float))
        out = np.empty((len(self.spheres), 3))
        for k, (s, j) in enumerate(zip(self.spheres, self.jid)):
            M = self.data.oMi[j]
            out[k] = M.rotation @ np.asarray(s[3]) + M.translation
        return out

    def clearances(self, q):
        """Per-pair (dist - threshold) in metres; negative = violated."""
        c = self.centers(q)
        out = np.empty(len(self.pairs))
        for k, (kind, i, j, thr) in enumerate(self.pairs):
            if kind == "body":
                cap = self.capsules[j]
                d = np.sqrt(_seg_dist2_np(c[i], np.asarray(cap[1]), np.asarray(cap[2])))
            else:
                d = float(np.linalg.norm(c[i] - c[j]))
            out[k] = d - thr
        return out

    def collides(self, q) -> bool:
        return bool(np.min(self.clearances(q)) < 0.0)

    def violated(self, q, tol=0.0):
        cl = self.clearances(q)
        return [(self.describe(k), float(cl[k])) for k in np.nonzero(cl < -tol)[0]]

    def describe(self, k):
        kind, i, j, _ = self.pairs[k]
        a = self.spheres[i]
        b = self.capsules[j][0] if kind == "body" else "{}:{}".format(self.spheres[j][0], self.spheres[j][2])
        return f"{a[0]}:{a[2]}#{i} vs {b}"

    def casadi_constraints(self, cdata, jid_of, aggregate=True, beta=SOFTMIN_BETA):
        """Symbolic g(q) >= 0 from a cpin data whose FK was computed on SX q.

        jid_of(joint_name) -> joint id in the cpin model (the IK's own reduced model).
        Per pair: c = (d^2 - thr^2) / (2 thr)  (~ d - thr in metres near contact, C1, no sqrt).
        aggregate=False: one constraint per pair (~800: IPOPT ~45 ms/solve).
        aggregate=True: one constraint per (arm sphere, body) and per (left sphere, right arm):
        softmin_beta(c) = m - log(sum exp(-beta (c_k - m))) / beta, m = min c_k. softmin <= min, so it is
        conservative (by at most log(n)/beta, typically << 1 mm since only 1-2 terms are near), smooth,
        and cuts the constraint count ~13x."""
        import casadi

        C = []
        for s in self.spheres:
            M = cdata.oMi[jid_of(s[1])]
            C.append(M.translation + M.rotation @ casadi.DM(np.asarray(s[3], float)))
        groups: dict = {}
        for kind, i, j, thr in self.pairs:
            if kind == "body":
                cap = self.capsules[j]
                a = np.asarray(cap[1], float)
                ab = np.asarray(cap[2], float) - a
                L2 = max(float(ab @ ab), 1e-12)
                t = casadi.fmax(0.0, casadi.fmin(1.0, casadi.dot(C[i] - a, casadi.DM(ab)) / L2))
                d2 = casadi.sumsqr(C[i] - a - t * casadi.DM(ab))
            else:
                d2 = casadi.sumsqr(C[i] - C[j])
            groups.setdefault((kind, i), []).append((d2 - thr * thr) / (2.0 * thr))
        if not aggregate:
            return casadi.vertcat(*[c for v in groups.values() for c in v])
        g = []
        for v in groups.values():
            if len(v) == 1:
                g.append(v[0])
                continue
            c = casadi.vertcat(*v)
            m = casadi.mmin(c)
            g.append(m - casadi.log(casadi.sum1(casadi.exp(-beta * (c - m)))) / beta)
        return casadi.vertcat(*g)
