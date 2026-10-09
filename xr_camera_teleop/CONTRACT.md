# xr_camera_teleop — interface contract

Goal: WiLoR (monocular RGB hand reconstruction) replaces the VR headset input of
`third_party/xr_teleoperate` (pinned 817fb00). Three outputs:

1. RGB overlay (served by the API server, `api/overlay.py`).
2. MuJoCo "floating hands": human wrists + 21 keypoints per hand in 3D.
3. G1 retargeting via xr_teleoperate's own `G1_29_ArmIK` (Pinocchio/CasADi) and
   dex-retargeting (Dex3), RELATIVE (clutch) mapping, MuJoCo G1 standing.

The WiLoR model (code + weights) is NOT part of this repo: `api/engine.py` loads it from `$WILOR_ROOT`
(default `third_party/WiLoR`, git-ignored). Camera helpers copied from the original `wilor_teleop/` live in
`xr_camera_teleop/camera/`.

## Environments

| Env | Path | Used by |
| --- | --- | --- |
| Perception (torch, WiLoR, pyzed) | `data/wilor-venv` | `xr_camera_teleop/api/server.py` only |
| Consumers (py3.10, mujoco 3.3, pinocchio 3.1 + casadi, numpy 1.26) | `data/xr-env` (micromamba) | everything else |

Add packages to `data/xr-env` with `data/xr-env/bin/python -m pip install ...`.

## Wire format (perception -> consumers)

ZMQ PUB, `tcp://127.0.0.1:5590`, single-part msgpack messages (no topic frame),
consumers use SUB with CONFLATE=1 (latest only). One message per processed frame:

```text
{
  "v": 1,
  "frame_id": int,
  "t_capture": float,      # time.monotonic() at frame dequeue, seconds
  "t_publish": float,      # time.monotonic() at publish
  "image_size": [w, h],
  "intrinsics": {"fx","fy","cx","cy"},
  "source": "live" | "replay",
  "max_hand_depth": float,  # OPTIONAL. Hands with wrist z beyond this (m) are not the operator's; overrides the
                            # consumers' 0.9 m default (Omron overhead camera: table distance + 0.05)
  "cam_R_level": [[3x3]],  # OPTIONAL. Camera levelling from the ZED IMU gravity (ZedSdkSource only):
                           # v_level = R @ v_cam (OpenCV axes), +y_level = gravity, yaw untouched
  "hands": {               # key present only when that hand is detected this frame
    "left"|"right": {
      "conf": float,                 # detector box confidence
      "wrist_pos": [3],              # metres, CAMERA frame (OpenCV: +x right, +y down, +z forward)
      "wrist_rot": [[3x3]],          # proper SO(3), camera frame, MANO root orientation
                                     # (left hand: handedness mirror already resolved, det=+1)
      "keypoints": [[21x3]],         # metres, CAMERA frame, MANO/OpenPose order, index 0 = wrist
      "keypoints_2d": [[21x2]]       # full-image pixels
    }
  }
}
```

MANO/OpenPose 21 order: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16;
pinky 17-20 (each finger: MCP/CMC, PIP, DIP, tip).

Python client (stdlib + pyzmq + msgpack only, importable from both envs):
`xr_camera_teleop/api/client.py` -> `HandStreamClient(addr).latest(max_age=0.3) -> dict | None`.

HTTP (same server, port 8770): `/` simple page, `/overlay.mjpg`, `/latest.json`, `/health`.

## Camera tilt

Consumers compute `p_W = R_W_CAM @ R_level @ p_cam` (same for rotations/keypoints) with
`R_level = msg["cam_R_level"]` when present, else a fixed pitch from the CLI `--cam-pitch-deg`
(positive = camera looking DOWN; `teleop/camlevel.py`, default 0 = level). The `--record-video`
`.json` sidecar carries `cam_R_level` (mean over the first second) when an IMU was available.

## Frames used downstream

- `CAM`: OpenCV camera frame above.
- `W`: viz/robot world, z up, x forward, y left. `R_W_CAM = [[0,0,1],[-1,0,0],[0,-1,0]]`
  (camera forward -> x, camera right -> -y, camera down -> -z). Camera treated
  as the operator's head, looking forward.
- xr_teleoperate expects `TeleData.left/right_wrist_pose` in robot basis, waist
  origin, Unitree arm URDF initial-pose convention, and `left/right_hand_pos`
  as (25,3) WebXR-ordered joints expressed in the wrist frame (see
  `teleop/televuer/src/televuer/tv_wrapper.py` for exact conventions — follow it).

## Relative (clutch) mapping

On calibrate (both hands seen, operator in a comfortable neutral):
`H_ref = human wrist pose (W)`, `R_ref = robot neutral wrist pose (waist)`.
Then each frame:
`p_robot = p_R_ref + k * (p_H - p_H_ref)` (k default 0.8, tunable, per-axis optional)
`R_robot = (R_H R_H_ref^T) R_R_ref` expressed in W, i.e. delta rotation applied in world.
Auto-calibration: both hands detected and nearly still (within 3 cm of their 1 s window mean) for 1.0 s.
A hand lost longer than the stale timeout (1 s) drifts back to the robot neutral (0.15 m/s, 0.8 rad/s)
and is re-anchored without a jump when it returns.
Camera placement therefore does not matter; only motion is transferred.
Handedness guard: once the operator heading is known (after calibration), with both hands seen the
"left" wrist must not be > 3 cm right of the "right" wrist (W +y = operator's left). Swapped for 0.15 s
-> stderr warning, both arms and hands return to the robot neutral until the sides are correct for 0.3 s;
calibration is postponed while the labels are swapped.
Fingers are not clutched (dex-retargeting uses hand-local vectors).

## Self-collision protection

`xrt_compat.make_arm_ik()` returns `teleop/safe_arm_ik.SafeG1_29_ArmIK` (subclass of the upstream `G1_29_ArmIK`,
same `solve_ik(left, right, q, dq) -> (q14, tauff14)`); `launch_xr_teleop.py` patches the upstream class name the
same way. Layer 1: sphere/capsule constraints in the IPOPT problem (`teleop/self_collision.SphereModel`, data in the
generated `teleop/g1_collision_spheres.py`, regenerate with `tools/fit_self_collision.py`). Layer 2: hpp-fcl check of
every output on the URDF collision meshes (`self_collision.MeshGate`); a colliding/NaN q is replaced by the last safe
q. Left-vs-right hand/wrist contact is allowed in both layers. Branch escape: a stuck warm-started branch is replaced
(per arm, smooth 2 rad/s joint-space transition) by a home-seeded solution when that is clearly better.
`ik.reset_to_home()` = rate-limited, collision-free joint-space return to q=0 + IK state reset; wired to
`WiLoRTeleWrapper.on_home_request` (called by `reset()`/`request_reset()` and the L/R swap latch) via
`xrt_compat.connect_ik(ik, wrapper)`; launcher: `xr_ctl.py reset` (SIGUSR1). Off switch: `--no-self-collision` /
`WILOR_XR_SELF_COLLISION=0` (`make_arm_ik(self_collision=False)` = upstream). On a solver failure the current motor q
is held (as upstream) if it is collision-free.

## Directory layout

```text
xr_camera_teleop/
  api/        server.py, client.py, replay.py        (agent A)
  viz/        floating_hands.py, hands_scene.py     (agent B)
  teleop/     wilor_tv_wrapper.py, mano_to_webxr.py, filters.py, clutch.py,
              g1_mujoco_teleop.py                   (agent C)
              safe_arm_ik.py, self_collision.py, g1_collision_spheres.py (generated)
  tests/      pure-numpy unit tests, run with data/xr-env/bin/python -m pytest
```
