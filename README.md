# xr-camera-teleoperation

Camera-based teleoperation of the Unitree G1 (29 DoF arms + Dex3 hands): a monocular RGB camera and the
WiLoR hand-reconstruction model replace the VR headset of Unitree's
[xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate).

```
camera ─► WiLoR server (GPU) ──ZMQ :5590──► clutch + filters ─► xr_teleoperate IK (+ self-collision) ─► DDS ─► G1
          xr_camera_teleop/api              xr_camera_teleop/teleop   third_party/xr_teleoperate (unmodified)
```

## Layout

| Path | What |
| --- | --- |
| `xr_camera_teleop/api/` | WiLoR hand server (ZMQ PUB :5590, HTTP :8770 overlay), client, replay/synthetic publisher |
| `xr_camera_teleop/camera/` | camera source helpers (ZED intrinsics, Omron FH, RGB, preprocessing) |
| `xr_camera_teleop/teleop/` | `TeleVuerWrapper` replacement, clutch, filters, MANO→WebXR, self-collision-safe arm IK |
| `xr_camera_teleop/launch_xr_teleop.py` | runs upstream `teleop_hand_and_arm.py` with the camera input (sim and real robot) |
| `xr_camera_teleop/ui/live_ui.py` | live web UI (:8780): overlay, floating hands, G1 preview |
| `xr_camera_teleop/sim/` | MuJoCo G1 DDS simulator (loopback), latency probe |
| `xr_camera_teleop/viz/`, `tools/`, `tests/` | visualisation, offline tools, unit tests |
| `third_party/xr_teleoperate` | git submodule, Unitree upstream pinned at `817fb00`, **not edited** (patched at import time) |
| `configs/camera/` | camera calibration (Omron FH) |
| `envs/` | package lists of the two Python environments |

The xr_teleoperate code is never edited: the launcher swaps `televuer.TeleVuerWrapper`, `G1_29_ArmIK` and
`teleimager.ImageClient` at import time.

## Not in this repo

- **WiLoR model** (code + `pretrained_models/`, `mano_data/`): clone it anywhere and set `WILOR_ROOT`
  (default `third_party/WiLoR`, git-ignored). Only the perception server needs it.
- **Robot camera**: the teleop program does not subscribe to the robot head camera (watched outside this program).
  The upstream image client is kept and can be re-enabled with `--robot-camera` / `XRCAM_ROBOT_CAMERA=1`.
- `data/`: Python environments, recordings, IK cache, camera caches (all created at runtime).

## Setup

```bash
git clone --recurse-submodules <this repo>
# perception env (torch, WiLoR deps, pyzed): data/wilor-venv   -> envs/perception.freeze.txt
# consumer env  (py3.10, mujoco, pinocchio+casadi, unitree_sdk2py, cyclonedds 0.10.2): data/xr-env -> envs/xr-env.freeze.txt
```

## Run

See [`xr_camera_teleop/RUN.md`](xr_camera_teleop/RUN.md) (sim + real robot),
[`xr_camera_teleop/GUIDE.md`](xr_camera_teleop/GUIDE.md) (live UI, Turkish) and
[`xr_camera_teleop/CONTRACT.md`](xr_camera_teleop/CONTRACT.md) (wire format, frames, clutch, self-collision).
