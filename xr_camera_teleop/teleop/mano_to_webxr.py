"""MANO/OpenPose 21 -> WebXR 25 joint layout, wrist frame, and finger-state scalars.

Pure numpy. Coordinates are whatever frame the input keypoints are in (use a
right-handed, metric frame; the functions are frame-agnostic except where a
"world" argument is documented).

WebXR hand joint order (25), as consumed by xr_teleoperate / dex-retargeting:
    0  wrist
    1..4   thumb  : metacarpal, proximal, distal, tip
    5..9   index  : metacarpal, proximal, intermediate, distal, tip
    10..14 middle : (same 5)
    15..19 ring   : (same 5)
    20..24 pinky  : (same 5)
MANO/OpenPose order (21): 0 wrist; thumb 1-4 (CMC, MCP, IP, tip);
index 5-8, middle 9-12, ring 13-16, pinky 17-20 (each MCP, PIP, DIP, tip).

Only WebXR indices 0 (wrist), 4 (thumb tip), 9 (index tip), 14 (middle tip) are
used by the Dex3 DexPilot config (assets/unitree_hand/unitree_dex3.yml:
target_link_human_indices_dexpilot [[9,14,14,0,0,0],[4,4,9,4,9,14]]), so the
interpolated metacarpals do not influence the G1 retargeting; they only matter
for visualisation / other hands.
"""
from __future__ import annotations

import numpy as np

# Fraction of the wrist->MCP segment at which the (missing) finger metacarpal
# joint (= carpometacarpal joint in WebXR) is placed. A human middle
# metacarpal bone is ~6-7 cm and wrist-pivot->MCP is ~9-10 cm, so the CMC sits
# ~3 cm from the wrist, i.e. ~0.3 of the way to the MCP.
METACARPAL_FRAC = 0.3

WEBXR_NAMES = (
    ["wrist"]
    + [f"thumb-{j}" for j in ("metacarpal", "phalanx-proximal", "phalanx-distal", "tip")]
    + [f"{f}-finger-{j}" for f in ("index", "middle", "ring", "pinky")
       for j in ("metacarpal", "phalanx-proximal", "phalanx-intermediate",
                 "phalanx-distal", "tip")]
)
assert len(WEBXR_NAMES) == 25

# (webxr_index, mano_index) for joints that exist in both layouts
_DIRECT = [(0, 0)]
_DIRECT += [(1 + i, 1 + i) for i in range(4)]            # thumb 1..4 <- 1..4
for _f in range(4):                                       # index..pinky
    for _j in range(4):                                   # MCP, PIP, DIP, tip
        _DIRECT.append((5 + 5 * _f + 1 + _j, 5 + 4 * _f + _j))
# (webxr_index, mano_mcp_index) for interpolated metacarpals
_META = [(5 + 5 * f, 5 + 4 * f) for f in range(4)]

THUMB_TIP, INDEX_TIP, MIDDLE_TIP = 4, 9, 14   # WebXR indices


def mano21_to_webxr25(kp21, meta_frac: float = METACARPAL_FRAC):
    """(21,3) MANO/OpenPose -> (25,3) WebXR. Metacarpals interpolated."""
    kp = np.asarray(kp21, dtype=float)
    if kp.shape != (21, 3):
        raise ValueError(f"expected (21,3), got {kp.shape}")
    out = np.empty((25, 3))
    for w, m in _DIRECT:
        out[w] = kp[m]
    for w, m in _META:
        out[w] = kp[0] + meta_frac * (kp[m] - kp[0])
    return out


def keypoint_wrist_frame(kp21, side: str):
    """Right-handed wrist frame from keypoints, OpenXR hand-tracking convention.

    Returns (R, origin): columns of R are, in the keypoint frame,
      x : wrist -> middle MCP (orthogonalised against z)
      z : back of hand (palm normal pointing away from the palm)
      y : z cross x  (= index->pinky for the left hand, pinky->index for the
          right hand, matching tv_wrapper.py docstring lines 11-19)
    Palm normal: left  hand z ~ (index_mcp-w) x (pinky_mcp-w);
                 right hand z ~ (pinky_mcp-w) x (index_mcp-w)
    (sign verified by test: left hand palm-down, fingers forward -> z up).
    """
    kp = np.asarray(kp21, dtype=float)
    w = kp[0]
    v_idx, v_mid, v_pky = kp[5] - w, kp[9] - w, kp[17] - w
    n = np.cross(v_idx, v_pky) if side == "left" else np.cross(v_pky, v_idx)
    # use ring MCP too for robustness: average of two normals
    v_rng = kp[13] - w
    n2 = np.cross(v_mid, v_pky) if side == "left" else np.cross(v_pky, v_mid)
    n3 = np.cross(v_idx, v_rng) if side == "left" else np.cross(v_rng, v_idx)
    # normalise each so that no single (near-degenerate) pair dominates
    nn = [x / (np.linalg.norm(x) + 1e-12) for x in (n, n2, n3)]
    z = nn[0] + nn[1] + nn[2]
    z /= np.linalg.norm(z) + 1e-12
    x = v_mid - np.dot(v_mid, z) * z
    x /= np.linalg.norm(x) + 1e-12
    y = np.cross(z, x)
    return np.column_stack([x, y, z]), w.copy()


def frame_angle(Ra, Rb):
    """Geodesic angle (rad) between two rotations."""
    c = (np.trace(np.asarray(Ra).T @ np.asarray(Rb)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def compare_with_wilor_rot(kp21, wrist_rot, side: str):
    """Compare keypoint-derived wrist frame with WiLoR's MANO root rotation.

    MANO's global_orient frame is not the OpenXR wrist frame: the two differ
    by a constant rotation C (R_mano = R_kp @ C). Returns (R_kp, C, residual
    angle of C in rad). Feeding many frames and checking that C is constant
    (low spread) tells which source is self-consistent; see xr_camera_teleop/teleop/
    README notes in wilor_tv_wrapper.py (we use the keypoint frame).
    """
    R_kp, _ = keypoint_wrist_frame(kp21, side)
    C = R_kp.T @ np.asarray(wrist_rot, float)
    return R_kp, C, frame_angle(C, np.eye(3))


def pinch_distance(webxr25):
    """Thumb-tip to index-tip distance (same unit as input)."""
    return float(np.linalg.norm(webxr25[THUMB_TIP] - webxr25[INDEX_TIP]))


def squeeze_value(webxr25) -> float:
    """Heuristic fist degree in [0,1] from fingertip-wrist distances.

    Ratio r = |tip - wrist| / |middle_mcp - wrist| averaged over index,
    middle, ring, pinky. Open flat hand r ~ 1.9-2.1; fist r ~ 0.9-1.1.
    Linearly mapped, clipped.
    """
    w = webxr25[0]
    ref = np.linalg.norm(webxr25[10 + 1] - w) + 1e-9      # middle proximal (MCP)
    tips = webxr25[[9, 14, 19, 24]]
    r = np.mean(np.linalg.norm(tips - w, axis=1)) / ref
    return float(np.clip((2.0 - r) / (2.0 - 1.0), 0.0, 1.0))
