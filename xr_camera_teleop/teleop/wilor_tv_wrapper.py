"""WiLoRTeleWrapper: drop-in replacement for TeleVuerWrapper.get_tele_data().

Pipeline per processed frame (camera-frame msg from xr_camera_teleop/api, CONTRACT.md):

 1. keypoints CAM -> W      p_W = R_W_CAM p_cam           (z up, x fwd, y left)
 2. 21 -> 25 WebXR joints   mano_to_webxr.mano21_to_webxr25
 3. wrist frame (R_W, p_W)  from KEYPOINTS (see "wrist source" below)
 4. One-Euro (pos) + SO(3) One-Euro (rot) on the wrist pose; One-Euro on the
    wrist-local (25,3) finger positions
 5. clutch (clutch.HandClutch): relative mapping into the robot waist frame.
    Output wrist pose is directly in G1_29_ArmIK's L_ee/R_ee frame convention,
    because the robot neutral comes from the IK model itself; so the
    tv_wrapper "T_TO_UNITREE_HUMANOID_*_ARM" right-multiply (tv_wrapper.py:
    332-333) is replaced by the neutral pose (only the human *delta* matters).
    The waist offset (+0.15 x, +0.45 z; tv_wrapper.py:117-118) is likewise
    already inside the robot neutral (IK frame origin = waist).
 6. velocity limiter on the output wrist pose (linear/angular).
 7. hand_pos (25,3): wrist-local XR-arm-convention coordinates (x wrist->middle,
    z back of hand; exactly inv(IPxr_arm) @ pos, tv_wrapper.py:362-363), then
    T_TO_UNITREE_HAND (tv_wrapper.py:132-135, applied at :375-376).
    hand_pos[0] == 0 (wrist). Dex3 DexPilot only reads joints 0,4,9,14.

Mapping (default "ego"): the camera is egocentric (operator's head/chest), so both hands
are anchored to ONE shared reference: p_robot = mid(neutral_L, neutral_R) + k (p_h - mid(p_hL0, p_hR0)),
which keeps the inter-hand geometry (hands together -> robot hands together), and the wrist
orientation is absolute: R_robot = R_h_W @ C_side, C_side maps the keypoint wrist frame of a
"fingers forward, palms facing in, thumbs up" hand onto the robot neutral (R ~ I).
mapping="per_hand" restores the old independent per-hand clutch (delta rotation).
Hands farther than max_depth from the camera are ignored (not the operator's).
Camera yaw (auto_yaw): at calibration the operator's forward is taken as the mean horizontal
wrist->middle-MCP direction of both hands (hands rest in front, fingers forward), so the camera
may face any way (egocentric, over the shoulder, across the table facing the operator).
cam_yaw_deg fixes it manually instead. IMU levelling gives pitch/roll only, never yaw.

Handedness guard: the detector sometimes swaps the left/right labels (edge cases, hands near each
other). Once the operator heading is known (after calibration, or a fixed cam_yaw_deg), the hand
labelled "left" must be on the operator's left (+y in W) whenever both hands are seen. If the left
wrist is more than swap_margin to the right of the right wrist for swap_hold seconds, a warning is
printed and both arms/hands return to the robot neutral until the sides are correct again for
swap_clear seconds (with a joint-space home hooked into on_home_request, e.g. SafeG1_29_ArmIK.reset_to_home
via xrt_compat.connect_ik, the arms are driven back to q=0 in joint space).
A calibration request with swapped labels is postponed (warning).

Wrist source: keypoint frame (x = wrist->middle-MCP, z = palm normal from
index/pinky/ring MCPs, y = z cross x) is used for BOTH the clutch rotation and
the hand_pos frame. Reasons: (a) it is by construction the OpenXR hand-tracking
convention that tv_wrapper/Dex3 expect, and is self-consistent with the
keypoints the finger vectors come from, so wrist->tip vectors stay correct even
when the MANO root regression drifts; (b) WiLoR's wrist_rot is the MANO root
(global_orient) frame, a different, unspecified-per-model axis convention that
would need an extra calibrated constant offset C (see
mano_to_webxr.compare_with_wilor_rot). Use `wrist_source="mano"` only for
experimentation (applies C estimated on the first valid frame).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from . import xrt_compat
from .camlevel import level_rotation, yaw_in_cam
from .clutch import HandClutch, StillDetector
from .filters import OneEuroFilter, PoseOneEuro, PoseRateLimiter
from .mano_to_webxr import (compare_with_wilor_rot, keypoint_wrist_frame, mano21_to_webxr25,
                            pinch_distance, squeeze_value)

R_W_CAM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
SIDES = ("left", "right")
PINCH_THRESH_M = 0.025
SQUEEZE_THRESH = 0.7

# FK of G1_29_ArmIK at q=0 (see clutch.py); used only if no IK-derived neutral is passed.
_NEUTRAL_FALLBACK = {"left": np.array([0.2498, 0.1487, 0.0952]),
                     "right": np.array([0.2498, -0.1486, 0.0952])}


def fallback_neutral():
    out = []
    for s in SIDES:
        T = np.eye(4)
        T[:3, 3] = _NEUTRAL_FALLBACK[s]
        out.append(T)
    return out


def ego_wrist_offset(side, R_neutral=np.eye(3)):
    """C with R_kp_neutral @ C = R_neutral, R_kp_neutral = keypoint wrist frame (W) of a hand with
    fingers forward (+x), palm facing the body midline, thumb up: x_kp = +x, z_kp (back of hand)
    = +y for the left / -y for the right hand, y_kp = z_kp x x_kp (keypoint_wrist_frame)."""
    z = np.array([0.0, 1.0, 0.0]) if side == "left" else np.array([0.0, -1.0, 0.0])
    x = np.array([1.0, 0.0, 0.0])
    R_kp = np.column_stack([x, np.cross(z, x), z])
    return R_kp.T @ np.asarray(R_neutral, float)


def _wrist_inside(entry, image_size, margin):
    """Wrist keypoint well inside the image (truncated corner detections give bad depth)."""
    kp2 = entry.get("keypoints_2d")
    if kp2 is None or image_size is None:
        return True
    u, v = kp2[0]
    w, h = image_size
    return margin <= u <= w - margin and margin <= v <= h - margin


class WiLoRTeleWrapper:
    def __init__(self, addr="tcp://127.0.0.1:5590", client=None, k=0.8, neutral=None,
                 stale_timeout=1.0, hand_timeout=0.25, max_age=0.3, use_filters=True,
                 max_lin=1.2, max_ang=4.0, hand_scale=1.0, wrist_source="keypoints",
                 auto_calibrate=False, clock=time.monotonic, cam_pitch_deg=0.0,
                 still_window=1.0, still_radius=0.03, mapping="ego", max_depth=0.9,
                 cal_margin_px=40, auto_yaw=True, cam_yaw_deg=None, swap_margin=0.03,
                 swap_hold=0.15, swap_clear=0.3):
        tv = xrt_compat.load_tv_wrapper()
        self._tv = tv
        self.TeleData = tv.TeleData
        if client is None:
            from xr_camera_teleop.api.client import HandStreamClient
            client = HandStreamClient(addr)
        self.client = client
        self.clock = clock
        self.max_age, self.hand_timeout = max_age, hand_timeout
        self.use_filters, self.hand_scale = use_filters, hand_scale
        self.wrist_source = wrist_source
        self.auto_calibrate = auto_calibrate
        self.cam_pitch_deg = cam_pitch_deg
        self._R_wc = R_W_CAM @ level_rotation(None, cam_pitch_deg)   # CAM -> W incl. tilt
        self._still = StillDetector(SIDES, still_window, still_radius, max_gap=hand_timeout)
        self.auto_cal_t = None
        assert mapping in ("ego", "per_hand")
        self.mapping, self.max_depth, self.cal_margin_px = mapping, max_depth, cal_margin_px
        self._cal_pending = False
        self._cal_ok = {s: False for s in SIDES}
        self.auto_yaw = auto_yaw and cam_yaw_deg is None
        self.yaw = 0.0 if cam_yaw_deg is None else np.radians(cam_yaw_deg)   # operator heading in level W
        self.swap_margin, self.swap_hold, self.swap_clear = swap_margin, swap_hold, swap_clear
        self._heading_known = not self.auto_yaw
        self.swapped = False                  # handedness conflict latched: robot held at neutral
        self._swap_t = None                   # since when the labels look swapped / correct again
        self._cal_swap_warned = False
        self._R_level = level_rotation(None, cam_pitch_deg)
        self._kp_cam = {s: None for s in SIDES}
        nl, nr = neutral if neutral is not None else fallback_neutral()
        if mapping == "ego":
            self.clutch = {s: HandClutch(n, k, stale_timeout, reanchor_on_stale=False,
                                         R_offset=ego_wrist_offset(s, n[:3, :3]))
                           for s, n in (("left", nl), ("right", nr))}
        else:
            self.clutch = {"left": HandClutch(nl, k, stale_timeout),
                           "right": HandClutch(nr, k, stale_timeout)}
        self._pose_f = {s: PoseOneEuro() for s in SIDES}
        self._finger_f = {s: OneEuroFilter(1.5, 1.0) for s in SIDES}
        self._limiter = {s: PoseRateLimiter(max_lin, max_ang) for s in SIDES}
        self._mano_C = {s: None for s in SIDES}
        self._last_fid = None
        self._last_proc_t = {s: None for s in SIDES}
        self._seen_t = {s: None for s in SIDES}
        self._ever_seen = {s: False for s in SIDES}
        self._pose = {s: None for s in SIDES}                # filtered (p_W, R_W)
        self._local25 = {s: None for s in SIDES}             # filtered wrist-local webxr25 (m)
        self._hand_pos_out = {s: self._default_hand_pos(s) for s in SIDES}
        self._pinch_m = {s: 10.0 for s in SIDES}
        self._squeeze = {s: 0.0 for s in SIDES}
        self._out_T = {s: self.clutch[s].neutral_T.copy() for s in SIDES}
        self.last_msg = None
        # called (no args) when the robot should return to its home posture: reset() and the handedness-swap
        # latch. Consumers with a joint-space home (SafeG1_29_ArmIK.reset_to_home) hook in here; without
        # it the clutch output simply goes to the neutral wrist poses as before.
        self.on_home_request = None
        self._reset_request = False
        self.stats = {"frames": 0}
        self.motion_data_ready = False
        T_RO = tv.T_ROBOT_OPENXR
        T_OR = np.linalg.inv(T_RO)
        self._head_pose = T_RO @ tv.CONST_HEAD_POSE @ T_OR   # constant, robot basis (tv_wrapper.py:324)

    # ---------------------------------------------------------------- control
    def calibrate(self, hands=SIDES):
        if self.mapping == "ego":       # shared anchor needs both hands: done on the next frame with both
            self._cal_pending = True
            return
        for s in hands:
            self.clutch[s].request_calibration()

    def request_reset(self):
        """Thread/signal-safe reset(): performed at the start of the next get_tele_data()."""
        self._reset_request = True

    def _home_request(self):
        cb = self.on_home_request
        if cb is not None:
            try:
                cb()
            except Exception as e:  # noqa: BLE001
                print(f"[xr_camera_teleop] on_home_request failed: {e!r}", file=sys.stderr, flush=True)

    def reset(self):
        """Uncalibrate: clutch outputs go to the robot neutral (recalibrate to resume), robot asked to go
        home in joint space (on_home_request)."""
        self._home_request()
        for s in SIDES:
            self.clutch[s].reset()
            self._pose_f[s].reset()
            self._finger_f[s].reset()
            self._limiter[s].reset()
        self._still.reset()
        self.auto_cal_t = None
        self._cal_pending = False
        self.swapped, self._swap_t = False, None
        if self.auto_yaw:                      # heading is re-estimated at the next calibration
            self.yaw = 0.0
            self._heading_known = False
            self._R_wc = R_W_CAM @ self.effective_level()

    def set_engaged(self, engaged: bool):
        for s in SIDES:
            self.clutch[s].set_engaged(engaged)

    def set_k(self, k):
        for s in SIDES:
            self.clutch[s].k = np.broadcast_to(np.asarray(k, float), (3,)).copy()

    def close(self):
        c = getattr(self.client, "close", None)
        if c:
            c()

    # ---------------------------------------------------------------- helpers
    def _default_hand_pos(self, side):
        from .synthetic_stream import canonical_hand_mano21
        return self._to_unitree_hand(mano21_to_webxr25(canonical_hand_mano21(side)))

    def _to_unitree_hand(self, local25):
        T3 = self._tv.T_TO_UNITREE_HAND[:3, :3]
        return local25 @ T3.T

    def _process_hand(self, side, entry, t):
        kp_cam = np.asarray(entry["keypoints"], float)
        if kp_cam.shape != (21, 3) or not np.all(np.isfinite(kp_cam)):
            return False
        max_depth = (self.last_msg or {}).get("max_hand_depth", self.max_depth) if self.max_depth else None
        if max_depth and kp_cam[0, 2] > max_depth:   # too far for the operator's own hand
            return False
        self._kp_cam[side] = kp_cam
        kpW = kp_cam @ self._R_wc.T
        R, p = keypoint_wrist_frame(kpW, side)
        if self.wrist_source == "mano" and entry.get("wrist_rot") is not None:
            Rm = self._R_wc @ np.asarray(entry["wrist_rot"], float)
            if self._mano_C[side] is None:
                self._mano_C[side] = R.T @ Rm
            R = Rm @ self._mano_C[side].T
        local = (mano21_to_webxr25(kpW) - p) @ R * self.hand_scale
        lp = self._last_proc_t[side]
        if lp is not None and t - lp > 0.5:
            self._pose_f[side].reset()
            self._finger_f[side].reset()
        self._last_proc_t[side] = t
        if self.use_filters:
            p, R = self._pose_f[side](p, R, t)
            local = self._finger_f[side](local, t)
        self._pose[side] = (p, R)
        self._local25[side] = local
        self._pinch_m[side] = pinch_distance(local)
        self._squeeze[side] = squeeze_value(local)
        self._hand_pos_out[side] = self._to_unitree_hand(local)
        return True

    def _ingest(self, now):
        msg = self.client.latest(max_age=self.max_age) if self.client else None
        if msg is None or msg.get("frame_id") == self._last_fid:
            return
        self._last_fid = msg.get("frame_id")
        self.last_msg = msg
        self.stats["frames"] += 1
        t = float(msg.get("t_capture", now))
        self._R_level = level_rotation(msg, self.cam_pitch_deg)
        self._R_wc = R_W_CAM @ self.effective_level()
        for s in SIDES:
            e = msg.get("hands", {}).get(s)
            if e is not None and self._process_hand(s, e, t):
                self._seen_t[s] = now
                self._ever_seen[s] = True
                self._still.push(s, t, self._pose[s][0])
                self._cal_ok[s] = _wrist_inside(e, msg.get("image_size"), self.cal_margin_px)
            else:
                self._cal_ok[s] = False
                self._still.lost(s)
        # auto-calibrate: both hands detected and nearly still for still_window seconds
        if self.auto_calibrate and self._still.ready() and all(self._cal_ok.values()):
            if self.mapping == "ego":
                if not self.clutch["left"].calibrated and not self._cal_pending:
                    self._cal_pending = True
                    if self.auto_cal_t is None:
                        self.auto_cal_t = now
                return
            for s in SIDES:
                if not self.clutch[s].calibrated:
                    self.clutch[s].request_calibration()
                    if self.auto_cal_t is None:
                        self.auto_cal_t = now

    def effective_level(self, msg=None):
        """yaw @ level in camera axes: consumers (viz HandMapper.R_level) use the same W as the clutch."""
        R_lv = self._R_level if msg is None else level_rotation(msg, self.cam_pitch_deg)
        return yaw_in_cam(-self.yaw) @ R_lv

    def _estimate_yaw(self):
        """Operator heading in level W from both hands' wrist->middle-MCP directions."""
        R0 = R_W_CAM @ self._R_level
        d = np.zeros(2)
        for s in SIDES:
            kp = self._kp_cam[s]
            v = R0 @ (kp[9] - kp[0])
            n = np.linalg.norm(v[:2])
            if n > 1e-6:
                d += v[:2] / n
        return None if np.linalg.norm(d) < 0.5 else float(np.arctan2(d[1], d[0]))

    def _set_yaw(self, yaw):
        """Change heading; already-filtered poses are rotated into the new W, filters restart."""
        Rz_d = R_W_CAM @ yaw_in_cam(-(yaw - self.yaw)) @ R_W_CAM.T
        self.yaw = yaw
        self._R_wc = R_W_CAM @ self.effective_level()
        for s in SIDES:
            if self._pose[s] is not None:
                p, R = self._pose[s]
                self._pose[s] = (Rz_d @ p, Rz_d @ R)
            self._pose_f[s].reset()
            self._limiter[s].reset()

    def _sides_swapped(self):
        """Hand labelled left is clearly on the operator's right (W: +y = operator's left)."""
        return self._pose["left"][0][1] < self._pose["right"][0][1] - self.swap_margin

    def _update_swap(self, now, both):
        """Debounced handedness conflict latch (see module docstring)."""
        bad = self._heading_known and both and self._sides_swapped()
        if bad == self.swapped:
            self._swap_t = None
            return
        if self._swap_t is None:
            self._swap_t = now
        if now - self._swap_t >= (self.swap_clear if self.swapped else self.swap_hold):
            self.swapped, self._swap_t = bad, None
            self.stats["swaps"] = self.stats.get("swaps", 0) + int(bad)
            if bad:
                self._home_request()
            print("[xr_camera_teleop] WARNING: left/right hand labels swapped (left hand on the right side); "
                  "robot returning to neutral" if bad else "[xr_camera_teleop] hand sides correct again: resuming",
                  file=sys.stderr, flush=True)

    # --------------------------------------------------------------- main API
    def get_tele_data(self):
        if self._reset_request:
            self._reset_request = False
            self.reset()
        now = self.clock()
        self._ingest(now)
        valid = {s: self._seen_t[s] is not None and (now - self._seen_t[s]) < self.hand_timeout for s in SIDES}
        both = all(valid.values())
        if self._cal_pending and both:
            if self.auto_yaw:
                y = self._estimate_yaw()
                if y is not None:
                    self._set_yaw(y)
                    self._heading_known = True
            if self._heading_known and self._sides_swapped():
                if not self._cal_swap_warned:
                    print("[xr_camera_teleop] WARNING: left/right hand labels swapped; calibration postponed",
                          file=sys.stderr, flush=True)
                    self._cal_swap_warned = True
            else:
                c_h = 0.5 * (self._pose["left"][0] + self._pose["right"][0])
                c_r = 0.5 * (self.clutch["left"].neutral_T[:3, 3] + self.clutch["right"].neutral_T[:3, 3])
                for s in SIDES:
                    p, R = self._pose[s]
                    self.clutch[s].calibrate(p, R, now, p_h_anchor=c_h, p_r_anchor=c_r)
                self._cal_pending = self._cal_swap_warned = False
        self._update_swap(now, both)
        for s in SIDES:
            if self.swapped:
                T = self.clutch[s].hold_neutral(now)
                self._hand_pos_out[s] = self._default_hand_pos(s)
            elif valid[s]:
                p, R = self._pose[s]
                T, _ = self.clutch[s].update(now, p, R, True)
            else:
                T, _ = self.clutch[s].update(now, valid=False)
            if self.use_filters:
                pl, Rl = self._limiter[s](T[:3, 3], T[:3, :3], now)
                T = np.eye(4)
                T[:3, :3], T[:3, 3] = Rl, pl
            self._out_T[s] = T
        self.motion_data_ready = any(self._ever_seen.values())
        L, R_ = "left", "right"
        return self.TeleData(
            head_pose=self._head_pose.copy(),
            left_wrist_pose=self._out_T[L].copy(),
            right_wrist_pose=self._out_T[R_].copy(),
            left_hand_pos=self._hand_pos_out[L].copy(),
            right_hand_pos=self._hand_pos_out[R_].copy(),
            left_hand_pinch=bool(self._pinch_m[L] < PINCH_THRESH_M),
            left_hand_pinchValue=float(self._pinch_m[L] * 100.0),   # cm, like tv_wrapper.py:408
            left_hand_squeeze=bool(self._squeeze[L] > SQUEEZE_THRESH),
            left_hand_squeezeValue=float(self._squeeze[L]),
            right_hand_pinch=bool(self._pinch_m[R_] < PINCH_THRESH_M),
            right_hand_pinchValue=float(self._pinch_m[R_] * 100.0),
            right_hand_squeeze=bool(self._squeeze[R_] > SQUEEZE_THRESH),
            right_hand_squeezeValue=float(self._squeeze[R_]),
            motion_data_ready=self.motion_data_ready,
        )

    def status(self):
        now = self.clock()
        out = {s: {"calibrated": self.clutch[s].calibrated,
                   "stale": self.clutch[s].stale,
                   "seen_age": None if self._seen_t[s] is None else now - self._seen_t[s]}
               for s in SIDES}
        out["swapped"] = self.swapped
        return out
