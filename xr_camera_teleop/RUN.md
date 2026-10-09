# Running WiLoR -> xr_teleoperate -> G1 (sim and real: same code path)

Canli web arayuzu (3 panel) ve Turkce kullanim rehberi: [GUIDE.md](GUIDE.md) (`xr_camera_teleop/ui/live_ui.py`, port 8780).

`xr_camera_teleop/launch_xr_teleop.py` runs the UNMODIFIED `third_party/xr_teleoperate/teleop/teleop_hand_and_arm.py`
with `televuer.TeleVuerWrapper` replaced by `WiLoRTeleWrapper` and the head-camera ImageClient replaced by a
no-camera stub. DDS, G1_29_ArmController (rt/lowcmd or rt/arm_sdk), Dex3 controller (rt/dex3/*), IK and
"go home on exit" are the original code. `--sim` only selects DDS domain 1 (real: domain 0).

Env: `PY=data/xr-env/bin/python` (has unitree_sdk2py, cyclonedds 0.10.2, sshkeyboard, rerun, mujoco, pinocchio).
Hand stream: `xr_camera_teleop/api/server.py` (live, wilor-venv) or `replay.py --synthetic` / `sim/latency_probe.py publish`.

## Simulation (loopback, domain 1)

```bash
# terminal 1: simulator (viewer window, 60 Hz passive viewer; add --headless for none)
$PY xr_camera_teleop/sim/g1_dds_mujoco.py --domain 1 --iface lo
# terminal 2: hand stream (live WiLoR server, or synthetic)
$PY xr_camera_teleop/api/replay.py --synthetic
# terminal 3: teleop (needs a real TTY for the keyboard)
$PY xr_camera_teleop/launch_xr_teleop.py --arm=G1_29 --ee=dex3 --input-mode=hand --sim --network-interface=lo
```
Keys (sshkeyboard, terminal 3): `r` start following (clutch calibrates on the first frame with both
hands, so hold a comfortable neutral, hands near the home pose, then press `r`), `q` stop: arms
go home, process exits. Without a TTY add `--ipc` and use `$PY xr_camera_teleop/xr_ctl.py start|stop|status`.
Our extra args (stripped before the original parser): `--addr tcp://127.0.0.1:5590` (or `WILOR_XR_ADDR`),
`--k 0.8` (`WILOR_XR_K`), `--no-auto-calibrate`, `--arm-vel-limit RAD_S`,
`--no-self-collision` (`WILOR_XR_SELF_COLLISION=0`; see "Self-collision protection" below).
Sim options: `--report-self-collision` (checks the arm q on the URDF meshes every 50 ms and prints contacts;
the MJCF has contype=0, so the sim itself never blocks them), `--elastic-band` (free base hung from a virtual spring instead of welded pelvis),
`--dt 0.0005`, `--view-hz 30`, `--trace f.npz`, `--summary f.json`, `--duration S`.
Debug-only kinematic tool (no DDS): `xr_camera_teleop/teleop/g1_mujoco_teleop.py`.

Notes: `--motion` in sim = rt/arm_sdk with the weight in motor_cmd[29].q; Unitree's locomotion/balance
controller is closed source, so the sim holds the legs (and un-blended joints) with a PD hold at q=0.
The sim prints RTF and step-time p50/p99 every 5 s. The `MotionSwitcher` call of the script (non-motion
mode) fails with "Failed" in sim (no service) - harmless.

## Real robot (domain 0)

```bash
ip a   # find the NIC on the robot network (192.168.123.x), e.g. enp3s0
$PY xr_camera_teleop/launch_xr_teleop.py --arm=G1_29 --ee=dex3 --input-mode=hand \
    --motion --network-interface=enp3s0 --arm-vel-limit 3.0
```
- `--motion`: arms via rt/arm_sdk blended with the robot's own controller (robot in regular "Regular mode", R1+X, not debug).
  Without `--motion` the script enters debug mode (releases the Unitree controller; the robot will NOT balance by itself:
  only with the robot hung/supported).
- Safety: the real path clips arm targets to `arm_velocity_limit` (default 30 rad/s - far too high for first tests):
  use `--arm-vel-limit 2..3`. Start with arms near home (q=0, arms hanging/neutral) because the clutch maps to the IK
  home pose and the first command is the neutral pose. Keep `--k` <= 0.8. Use the robot's e-stop / remote damping
  (L2+B on the remote) at any time; `q` makes arms go home then ramps the arm_sdk weight 1 -> 0.
- Dry run the same command first with the sim on domain 0 (loopback): `g1_dds_mujoco.py --domain 0 --iface lo`
  and `--network-interface=lo` on the launcher (do not do this with the real robot connected).

## Self-collision protection, branch escape, reset to home (default ON)

The upstream `G1_29_ArmIK` has no self-collision handling (arm into torso / hand into hand). Every IK user
(`launch_xr_teleop.py`, `ui/live_ui.py`, `teleop/g1_mujoco_teleop.py`, `tools/render_session.py`) now gets
`xr_camera_teleop/teleop/safe_arm_ik.SafeG1_29_ArmIK` (subclass; `third_party/` is untouched): `xrt_compat.make_arm_ik()`
returns it, and the launcher replaces `teleop.robot_control.robot_arm_ik.G1_29_ArmIK` before running the script
(stderr: `self-collision protection ON`).

1. **IK constraints**: arm links as 25 spheres per arm, body (torso 5 slabs, head, pelvis, hips/thighs) as 15 capsules,
   fitted to the URDF collision meshes (`tools/fit_self_collision.py` -> `teleop/g1_collision_spheres.py`, every mesh
   vertex covered); margin 5 mm vs body, 1 cm arm vs arm. 818 pairs (each arm vs body; left vs right arm EXCEPT
   hand/wrist vs hand/wrist: hands may touch / clap / rest wrist on wrist), soft-min aggregated into 73 smooth IPOPT
   inequalities. Fast path: upstream's unconstrained problem first; if its result satisfies the constraints it is the
   answer (~1.1 ms vs 0.8 ms upstream), otherwise the constrained problem (~11 ms median): the arm stops at the body
   (hand ~2 cm from the chest) instead of going through it.
2. **Post-solve gate**: each output is checked on the real meshes (hpp-fcl, ~0.1 ms; hand/wrist vs hand/wrist contact
   allowed). If upstream's 4-tap moving average cuts through the body, the first safe point toward the raw solution
   is used; colliding or non-finite q is never returned (last safe q + tauff, rate-limited stderr warning).
3. **Branch escape** (fixes "arms stay twisted after unusual hand motions"): the warm-started IK (seed = current q,
   `0.1|q-q_last|^2`, joint limits) stays on a joint-limit-pinned / wound branch when wrist-rotation targets leave the
   wrist range (+-113/92 deg) - also upstream (25/40 random excursions stuck). Every 0.25 s (backing off to 2 s while
   nothing better is found) when the tracking cost is not small, a solve seeded from home is compared per arm; if it
   is clearly better (< 0.5x cost and > 0.05 lower) the arm moves there in joint space at 2 rad/s, collision-free
   (straight line, else a collision-constrained projected step), following the operator on the new branch; then
   tracking resumes. 0/40 stuck with it.
4. **Reset to home**: `SafeG1_29_ArmIK.reset_to_home()` drives both arms to q=0 in joint space (2 rad/s, collision-free,
   targets ignored meanwhile), then clears warm start / filter / last safe q / escape state. Hooked to
   `wrapper.reset()` and to the L/R swap latch via `xrt_compat.connect_ik(ik, wrapper)`: live UI **Reset** button (R),
   `g1_mujoco_teleop.py`, `render_session.py`; launcher: `$PY xr_camera_teleop/xr_ctl.py reset` (SIGUSR1 via
   `data/xr_run/launch_xr_teleop.pid`, works with or without `--ipc`). Recalibrate afterwards (auto-calibration).

Timing (this machine): free space median 1.1 ms; in contact median ~11 ms, p95 ~25 ms; an escape check that needs a
cold constrained solve costs 20-55 ms (at most every 2 s while the target is held inside the body).
Disable (debug only): launcher `--no-self-collision`, `live_ui.py --no-self-collision`, or `WILOR_XR_SELF_COLLISION=0`
for everything (upstream IK: no escape, no joint-space reset). Tests: `$PY -m pytest -q xr_camera_teleop/tests/test_self_collision.py -s`.
The live UI status line shows `self-collision: ON ...` with constrained / gated / escapes / homing.
Limitations: the IK model has legs/waist locked at 0 and the Dex3 fingers straight; the upper arm may not come closer to
the torso than at q=0 (mesh gap there 8.6 mm); hands are NOT kept apart at all (they can also be pushed into each
other); escape/home transitions are joint-space moves the operator did not command (2 rad/s).
