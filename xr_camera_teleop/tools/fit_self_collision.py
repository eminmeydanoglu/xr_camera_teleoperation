#!/usr/bin/env python
"""Fit the IK self-collision primitives to the G1 URDF collision meshes -> teleop/g1_collision_spheres.py.

    data/xr-env/bin/python xr_camera_teleop/tools/fit_self_collision.py [--margin 0.01] [--check]

Arm links: per group (parent joint, list of meshes) k-means spheres (init along the PCA axis, Lloyd
iterations), radius = max distance of the cluster's vertices -> every vertex is covered.
Body: capsules with a fixed axis; for the given axis line (through the bbox centre) the radius is the
max radial distance and the end points are the tightest that still cover every vertex.
Pairs: every arm sphere vs every body capsule (except shoulder cylinders), left vs right arm spheres;
pairs that never get within 5 cm of their threshold over random joint samples are dropped;
pairs closer than threshold + HOME_SLACK at q=0 get threshold = home distance - HOME_SLACK.
"""
from __future__ import annotations

import argparse
import warnings
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from xr_camera_teleop.teleop.self_collision import SIDES, build_reduced_models  # noqa: E402

OUT = ROOT / "xr_camera_teleop" / "teleop" / "g1_collision_spheres.py"
HOME_SLACK = 0.005

# (group, parent joint suffix, mesh link suffixes, n spheres)
ARM_GROUPS = [
    ("shoulder", "shoulder_roll_joint", ["shoulder_roll_link"], 1),
    ("upper_arm", "shoulder_yaw_joint", ["shoulder_yaw_link"], 3),
    ("elbow", "elbow_joint", ["elbow_link"], 3),
    ("wrist_roll", "wrist_roll_joint", ["wrist_roll_link"], 2),
    ("wrist_pitch", "wrist_pitch_joint", ["wrist_pitch_link"], 2),
    ("hand", "wrist_yaw_joint", ["wrist_yaw_link", "hand_palm_link", "hand_index_0_link", "hand_index_1_link",
                                  "hand_middle_0_link", "hand_middle_1_link"], 10),
    ("thumb", "wrist_yaw_joint", ["hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link"], 4),
]
NOT_VS_BODY = ("shoulder",)
# Hands and wrists may touch each other (clapping, wrist on wrist): no left-vs-right pair when BOTH spheres
# are in these groups. Upper arm / elbow (forearm) vs anything of the other arm stays protected.
HAND_WRIST = ("wrist_roll", "wrist_pitch", "hand", "thumb")
# body capsules: (name, meshes, axis, slab axis/range or None)
BODY = [("torso_%d" % k, ["torso_link", "logo_link"], "y", ("z", z0, z1))
        for k, (z0, z1) in enumerate(zip(np.linspace(0.03, 0.36, 6)[:-1], np.linspace(0.03, 0.36, 6)[1:]))] + [
    ("head", ["head_link"], "z", ("z", 0.36, 0.6)),
    ("pelvis", ["pelvis_contour_link"], "y", None),
] + [(f"{s}_{l}", [f"{s}_{l}_link"], ax, None) for s in SIDES
     for l, ax in (("hip_pitch", "z"), ("hip_roll", "z"), ("hip_yaw", "z"), ("knee", "z"))]
AX = {"x": 0, "y": 1, "z": 2}


def geom_vertices(g, M=None):
    """Vertices of a geometry object in its parent-joint frame (meshes; boxes/cylinders sampled)."""
    import hppfcl

    m = g.geometry
    if isinstance(m, hppfcl.Box):
        h = np.asarray(m.halfSide)
        s = np.linspace(-1, 1, 7)
        P = np.array([[a, b, c] for a in s for b in s for c in s]) * h
    elif isinstance(m, hppfcl.Cylinder):
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        P = np.array([[m.radius * np.cos(t), m.radius * np.sin(t), z] for t in th
                      for z in np.linspace(-m.halfLength, m.halfLength, 7)])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # BVHModel.vertices(i) is deprecated in hpp-fcl 2.4
            P = np.array([m.vertices(j) for j in range(m.num_vertices)])
    M = g.placement if M is None else M
    return P @ M.rotation.T + M.translation


def _rup(r):
    """Radius rounded UP to 0.1 mm, + 0.1 mm for the 4-decimal rounding of the centres."""
    return float(np.ceil((r + 1e-4) * 1e4) / 1e4)


def kmeans_spheres(X, k, iters=60):
    c0 = X.mean(0)
    u = np.linalg.svd(X - c0, full_matrices=False)[2][0]
    t = (X - c0) @ u
    C = c0 + np.outer(np.linspace(np.percentile(t, 5), np.percentile(t, 95), k), u) if k > 1 else c0[None]
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([X[lab == i].mean(0) if np.any(lab == i) else C[i] for i in range(k)])
    # minimax refinement: move each centre toward its cluster's bbox centre (smaller max radius)
    lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
    out = []
    for i in range(k):
        P = X[lab == i]
        c = 0.5 * (P.min(0) + P.max(0))
        for _ in range(200):              # Badoiu-Clarkson minimum enclosing ball iteration
            far = P[np.argmax(((P - c) ** 2).sum(1))]
            c = c + (far - c) * 0.05
        r = float(np.sqrt(((P - c) ** 2).sum(1).max()))
        out.append((c, r))
    return out


def fit_capsule(X, axis):
    a = AX[axis]
    perp = [i for i in range(3) if i != a]
    ctr = 0.5 * (X.min(0) + X.max(0))
    rad = np.sqrt(((X[:, perp] - ctr[perp]) ** 2).sum(1))
    r = float(rad.max())
    h = np.sqrt(np.maximum(r * r - rad * rad, 0.0))
    t = X[:, a]
    lo, hi = float((t + h).min()), float((t - h).max())
    if lo > hi:
        lo = hi = 0.5 * (lo + hi)
    A, B = ctr.copy(), ctr.copy()
    A[a], B[a] = lo, hi
    return A, B, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin-body", type=float, default=0.005, help="clearance arm vs body [m]")
    ap.add_argument("--margin-arm", type=float, default=0.01, help="clearance left arm vs right arm [m]")
    ap.add_argument("--samples", type=int, default=50000)
    args = ap.parse_args()
    import pinocchio as pin

    red, geom = build_reduced_models()
    by_name = {g.name: g for g in geom.geometryObjects}

    spheres = []          # (side, joint, group, c, r)
    for s in SIDES:
        for grp, jn, links, k in ARM_GROUPS:
            jname = f"{s}_{jn}"
            jid = red.getJointId(jname)
            X = []
            for l in links:
                g = by_name[f"{s}_{l}_0"]
                assert g.parentJoint == jid, (g.name, red.names[g.parentJoint], jname)
                X.append(geom_vertices(g))
            X = np.concatenate(X)
            for c, r in kmeans_spheres(X, k):
                spheres.append((s, jname, grp, np.round(c, 4), _rup(r)))
    capsules = []
    for name, links, axis, slab in BODY:
        X = np.concatenate([geom_vertices(by_name[f"{l}_0"]) for l in links])
        if slab is not None:
            ax, z0, z1 = slab
            X = X[(X[:, AX[ax]] >= z0 - 1e-9) & (X[:, AX[ax]] <= z1 + 1e-9)]
        if len(X) == 0:
            continue
        A, B, r = fit_capsule(X, axis)
        capsules.append((name, np.round(A, 4), np.round(B, 4), _rup(r)))

    # candidate pairs, thresholds
    MB, MA = args.margin_body, args.margin_arm
    cand = []
    for i, sp in enumerate(spheres):
        if sp[2] not in NOT_VS_BODY:
            for j, cap in enumerate(capsules):
                cand.append(("body", i, j, sp[4] + cap[3] + MB))
    for i, a in enumerate(spheres):
        for j, b in enumerate(spheres):
            if a[0] == "left" and b[0] == "right" and not (a[2] in HAND_WRIST and b[2] in HAND_WRIST):
                cand.append(("arm", i, j, a[4] + b[4] + MA))

    data = red.createData()
    jids = [red.getJointId(s[1]) for s in spheres]

    def centers(q):
        pin.forwardKinematics(red, data, q)
        return np.array([data.oMi[j].rotation @ s[3] + data.oMi[j].translation for s, j in zip(spheres, jids)])

    def dists(C):
        out = np.empty(len(cand))
        for k, (kind, i, j, _) in enumerate(cand):
            if kind == "body":
                _, A, B, _ = capsules[j]
                ab = B - A
                t = np.clip(np.dot(C[i] - A, ab) / max(ab @ ab, 1e-12), 0, 1)
                out[k] = np.linalg.norm(C[i] - A - t * ab)
            else:
                out[k] = np.linalg.norm(C[i] - C[j])
        return out

    thr = np.array([c[3] for c in cand])
    d_home = dists(centers(np.zeros(red.nq)))
    # reachability pruning by random sampling inside the joint limits
    rng = np.random.default_rng(0)
    lo, hi = red.lowerPositionLimit, red.upperPositionLimit
    min_cl = d_home - thr
    for _ in range(args.samples):
        q = rng.uniform(lo, hi)
        min_cl = np.minimum(min_cl, _fast(centers(q), cand, capsules) - thr)
    keep = min_cl < 0.05
    relaxed = []
    pairs = []
    for k, c in enumerate(cand):
        if not keep[k]:
            continue
        t = thr[k]
        if d_home[k] < t + HOME_SLACK:
            t = d_home[k] - HOME_SLACK
            relaxed.append((c, thr[k], t))
        pairs.append((c[0], c[1], c[2], float(t)))

    def fmt(v):
        return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"

    L = ['"""GENERATED by xr_camera_teleop/tools/fit_self_collision.py - do not edit by hand (re-run the tool).',
         "",
         "Self-collision primitives fitted to the URDF collision meshes of",
         "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf (14-dof arm-reduced model, q=0 lock).",
         "ARM_SPHERES: (side, parent joint, group, centre in the parent JOINT frame [m], radius [m]).",
         "BODY_CAPSULES: (name, end A, end B, radius) in the pelvis frame (= IK base frame).",
         "PAIRS: (kind, sphere i, capsule j | sphere j, threshold [m]) - constraint dist >= threshold.",
         "threshold = r_i + r_j + MARGIN[kind], except {len(relaxed)} pairs already closer at q=0",
         f"(threshold = home distance - {HOME_SLACK}). {int((~keep).sum())} of {len(cand)} candidate pairs",
         f"were dropped as unreachable (never within 5 cm of the threshold over {args.samples} random q).",
         "Left-vs-right pairs between two hand/wrist spheres are excluded on purpose (hands may touch).",
         '"""', "", f"MARGIN = {{'body': {MB}, 'arm': {MA}}}", f"HAND_WRIST_GROUPS = {HAND_WRIST!r}", f"HOME_SLACK = {HOME_SLACK}", "", "ARM_SPHERES = ["]
    for s, jn, grp, c, r in spheres:
        L.append(f'    ("{s}", "{jn}", "{grp}", {fmt(c)}, {r:.4f}),')
    L += ["]", "", "BODY_CAPSULES = ["]
    for name, A, B, r in capsules:
        L.append(f'    ("{name}", {fmt(A)}, {fmt(B)}, {r:.4f}),')
    L += ["]", "", "# relaxed at home: " + "; ".join(
        f"{c[0]} {c[1]}-{c[2]} {a:.3f}->{b:.3f}" for c, a, b in relaxed), "PAIRS = ["]
    for kind, i, j, t in pairs:
        L.append(f'    ("{kind}", {i}, {j}, {t:.4f}),')
    L += ["]", ""]
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT}: {len(spheres)} arm spheres, {len(capsules)} body capsules, {len(pairs)} pairs "
          f"({len(relaxed)} relaxed at home, {int((~keep).sum())} pruned)")
    for s, jn, grp, c, r in spheres:
        print(f"  sphere {s:5s} {grp:11s} r={r:.3f} c={np.round(c, 3)}")
    for name, A, B, r in capsules:
        print(f"  capsule {name:16s} r={r:.3f} A={np.round(A, 3)} B={np.round(B, 3)}")
    for c, a, b in relaxed:
        print(f"  relaxed {c[0]} {spheres[c[1]][0]}:{spheres[c[1]][2]} vs "
              f"{capsules[c[2]][0] if c[0] == 'body' else spheres[c[2]][2]}: {a:.3f} -> {b:.3f}")


_CACHE = None


def _fast(C, cand, capsules):
    """Vectorised version of dists() for the candidate list (index arrays cached per list)."""
    global _CACHE
    if _CACHE is None or _CACHE[0] is not cand:
        bi = np.array([k for k, c in enumerate(cand) if c[0] == "body"])
        ai = np.array([k for k, c in enumerate(cand) if c[0] == "arm"])
        bs = np.array([cand[k][1] for k in bi])
        A = np.array([capsules[cand[k][2]][1] for k in bi])
        B = np.array([capsules[cand[k][2]][2] for k in bi])
        a1 = np.array([cand[k][1] for k in ai])
        a2 = np.array([cand[k][2] for k in ai])
        _CACHE = (cand, bi, ai, bs, A, B, a1, a2)
    _, bi, ai, bs, A, B, a1, a2 = _CACHE
    out = np.empty(len(cand))
    P = C[bs]
    ab = B - A
    t = np.clip(((P - A) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1)
    out[bi] = np.linalg.norm(P - A - t[:, None] * ab, axis=1)
    out[ai] = np.linalg.norm(C[a1] - C[a2], axis=1)
    return out


if __name__ == "__main__":
    main()
