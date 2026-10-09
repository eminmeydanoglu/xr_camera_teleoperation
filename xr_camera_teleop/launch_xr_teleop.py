#!/usr/bin/env python
"""Run xr_teleoperate's UNMODIFIED teleop_hand_and_arm.py with WiLoR as the input device.

Only the input changes: `televuer.TeleVuerWrapper` is replaced by `WiLoRTeleWrapper`
(returns the same TeleData) and the head-camera `teleimager.ImageClient` is replaced by a
no-camera stub (no image server; the robot camera is watched outside this program, see
--robot-camera). Everything else - DDS init (`--sim` => domain 1, else domain 0),
G1_29_ArmController (rt/lowcmd or rt/arm_sdk with --motion), Dex3 controller, IK, 'go home on exit' - is the original code, identical for sim and real robot.

    data/xr-env/bin/python xr_camera_teleop/launch_xr_teleop.py [our args] [teleop_hand_and_arm.py args]

Our args (stripped before the original script sees argv; env var in brackets):
    --addr tcp://127.0.0.1:5590   WiLoR stream address                 [WILOR_XR_ADDR]
    --k 0.8                       clutch gain human->robot motion      [WILOR_XR_K]
    --no-auto-calibrate           don't calibrate clutch on first frame with both hands
    --arm-vel-limit RAD_S         arm joint velocity limit applied by G1_29_ArmController
                                  (real-robot path only: the controller skips the limit
                                  when --sim). Default: unchanged (30 rad/s).
    --no-self-collision           use the original G1_29_ArmIK (no self-collision constraints
                                  and no mesh gate)                    [WILOR_XR_SELF_COLLISION=0]
    --robot-camera                DISABLED BY DEFAULT. Use upstream's real teleimager ImageClient
                                  (subscribes to the robot head camera at --img-server-ip; needs the
                                  teleimager package and its server on PC2). Off = no-camera stub,
                                  the program never subscribes to any robot image stream.
                                                                       [XRCAM_ROBOT_CAMERA=1]
    By default G1_29_ArmIK is replaced by xr_camera_teleop/teleop/safe_arm_ik.SafeG1_29_ArmIK
    (self-collision constraints inside the IK + real-mesh post-solve gate).
Everything else (--arm=G1_29 --ee=dex3 --input-mode=hand --sim --motion
--network-interface=eth0 --ipc --frequency ...) is passed through verbatim.

Start/stop: the script waits for key 'r' (sshkeyboard, needs a real TTY) to start following,
'q' to stop (arms go home, then exit). Without a TTY use `--ipc` and
`data/xr-env/bin/python xr_camera_teleop/xr_ctl.py start|stop`.
Reset (any mode): `data/xr-env/bin/python xr_camera_teleop/xr_ctl.py reset` (SIGUSR1 via the pid file
data/xr_run/launch_xr_teleop.pid): clutch uncalibrated, arms back to q=0 in joint space, then recalibrate.
"""
from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
XRT = REPO / "third_party" / "xr_teleoperate"
SCRIPT = XRT / "teleop" / "teleop_hand_and_arm.py"
RUN_DIR = REPO / "data" / "xr_run" / "teleop"     # has ../assets symlink + IK model cache


def _strip_own_args(argv):
    """Return (own_cfg, passthrough_argv). Accepts `--x v` and `--x=v`."""
    cfg = {"addr": os.environ.get("WILOR_XR_ADDR", "tcp://127.0.0.1:5590"),
           "k": float(os.environ.get("WILOR_XR_K", "0.8")),
           "auto_calibrate": True, "arm_vel_limit": None,
           "cam_pitch_deg": float(os.environ.get("WILOR_XR_CAM_PITCH", "0")),
           "self_collision": os.environ.get("WILOR_XR_SELF_COLLISION", "1").strip().lower()
           not in ("0", "false", "no", "off"),
           "robot_camera": os.environ.get("XRCAM_ROBOT_CAMERA", "0").strip().lower()
           in ("1", "true", "yes", "on")}
    valued = {"--addr": "addr", "--k": "k", "--arm-vel-limit": "arm_vel_limit",
              "--cam-pitch-deg": "cam_pitch_deg"}
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        name, eq, val = a.partition("=")
        if name in valued:
            if not eq:
                i += 1
                val = argv[i]
            cfg[valued[name]] = val
        elif a == "--no-auto-calibrate":
            cfg["auto_calibrate"] = False
        elif a == "--no-self-collision":
            cfg["self_collision"] = False
        elif a == "--robot-camera":
            cfg["robot_camera"] = True
        else:
            out.append(a)
        i += 1
    cfg["k"] = float(cfg["k"])
    cfg["cam_pitch_deg"] = float(cfg["cam_pitch_deg"])
    if cfg["arm_vel_limit"] is not None:
        cfg["arm_vel_limit"] = float(cfg["arm_vel_limit"])
    return cfg, out


class _NoCameraImageClient:
    """Stand-in for teleimager.ImageClient: reports no cameras (no zmq/webrtc, no frames)."""

    def __init__(self, *a, **k):
        pass

    def get_cam_config(self):
        cam = {"enable_zmq": False, "enable_webrtc": False, "binocular": False,
               "image_shape": [480, 640], "fps": 30, "webrtc_port": 0}
        return {"head_camera": dict(cam), "left_wrist_camera": dict(cam), "right_wrist_camera": dict(cam)}

    def get_head_frame(self):
        return types.SimpleNamespace(bgr=None)

    get_left_wrist_frame = get_right_wrist_frame = get_head_frame

    def close(self):
        pass


def _install_stubs(cfg):
    sys.path.insert(0, str(REPO))
    from xr_camera_teleop.teleop.wilor_tv_wrapper import WiLoRTeleWrapper
    from xr_camera_teleop.teleop import xrt_compat
    from xr_camera_teleop.teleop.clutch import neutral_from_ik_fk

    # neutral robot wrist poses from the very IK model teleop_hand_and_arm.py will use
    # (ctrl_dual_arm_go_home drives to q=0, and so does startup of the IK).
    neutral = None
    try:
        neutral = neutral_from_ik_fk(xrt_compat.make_arm_ik(self_collision=False))   # FK only
    except Exception as e:      # fall back to the constants baked into the wrapper
        print(f"[launch_xr_teleop] IK neutral unavailable ({e}); using fallback neutral", file=sys.stderr)

    def TeleVuerWrapper(*args, **kwargs):          # same call signature as televuer's, args ignored
        w = WiLoRTeleWrapper(addr=cfg["addr"], k=cfg["k"], neutral=neutral,
                             auto_calibrate=cfg["auto_calibrate"],
                             cam_pitch_deg=cfg["cam_pitch_deg"])
        if cfg["self_collision"]:
            # reset / L-R swap -> arms back to q=0 in joint space (the IK is built later by the script)
            from xr_camera_teleop.teleop.safe_arm_ik import request_home_all
            w.on_home_request = request_home_all
        _WRAPPERS.append(w)
        return w

    tv = types.ModuleType("televuer")
    tv.TeleVuerWrapper = TeleVuerWrapper
    sys.modules["televuer"] = tv
    if cfg["robot_camera"]:          # upstream teleimager.ImageClient stays in place (opt-in)
        return
    ti = types.ModuleType("teleimager")
    ic = types.ModuleType("teleimager.image_client")
    ic.ImageClient = _NoCameraImageClient
    ti.image_client = ic
    sys.modules["teleimager"] = ti
    sys.modules["teleimager.image_client"] = ic


_WRAPPERS = []
PID_FILE = RUN_DIR.parent / "launch_xr_teleop.pid"   # data/xr_run/


def _install_reset_signal():
    """SIGUSR1 (`xr_ctl.py reset`) -> wrapper.reset() on the next frame: clutch uncalibrated (auto
    recalibrates, or press calibrate), arms driven back to the home posture q=0 in joint space."""
    import signal

    def on_usr1(signum, frame):
        for w in _WRAPPERS:
            w.request_reset()

    signal.signal(signal.SIGUSR1, on_usr1)
    try:
        PID_FILE.write_text(str(os.getpid()))
    except OSError:
        pass


def _patch_velocity_limit(limit):
    """Optional safety knob: lower the arm velocity limit used on the real-robot path."""
    sys.path.insert(0, str(XRT))
    os.chdir(RUN_DIR)
    import teleop.robot_control.robot_arm as ra
    for name in ("G1_29_ArmController", "G1_29_Arm_Internal_Dex1_Controller"):
        cls = getattr(ra, name)
        orig = cls.set_arm_velocity_limit

        def patched(self, velocity_limit=30.0, _orig=orig, _lim=limit):
            _orig(self, _lim)
        cls.set_arm_velocity_limit = patched


def main():
    cfg, rest = _strip_own_args(sys.argv[1:])
    # logging_mp may only be configured once, before any getLogger(); the script's own
    # basicConfig(INFO) call would raise after we imported wrapper/IK modules -> configure now, make it a no-op
    import logging_mp
    logging_mp.basicConfig(level=logging_mp.INFO)
    logging_mp.basicConfig = lambda *a, **k: None
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # sys.path[0] is xr_camera_teleop/ whose `teleop` package would shadow xr_teleoperate's `teleop`
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != str(Path(__file__).resolve().parent)]
    sys.path.insert(0, str(XRT))
    os.chdir(RUN_DIR)                 # script uses cwd-relative '../assets/...' and a model cache
    _install_stubs(cfg)
    print("[launch_xr_teleop] robot camera " + ("ON (teleimager ImageClient)" if cfg["robot_camera"]
          else "OFF (no image subscription; --robot-camera to enable)"), file=sys.stderr)
    _install_reset_signal()
    if cfg["arm_vel_limit"] is not None:
        _patch_velocity_limit(cfg["arm_vel_limit"])
    if cfg["self_collision"]:
        # the script does `from teleop.robot_control.robot_arm_ik import G1_29_ArmIK` -> gets the safe class
        from xr_camera_teleop.teleop.safe_arm_ik import install_into_upstream
        install_into_upstream()
        print("[launch_xr_teleop] self-collision protection ON (SafeG1_29_ArmIK)", file=sys.stderr)
    else:
        print("[launch_xr_teleop] WARNING: self-collision protection OFF (--no-self-collision)", file=sys.stderr)
    # IPC_Server.stop() calls zmq.Context.instance().term(), which blocks forever while our
    # HandStreamClient SUB socket (same shared context) is still open -> close sockets first.
    import zmq
    zmq.Context.term = lambda self: self.destroy(linger=0)
    sys.argv = [str(SCRIPT)] + rest
    code = 0
    try:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    except SystemExit as e:          # the script ends with exit(0) in its finally block
        code = e.code if isinstance(e.code, int) else 0
    finally:
        try:
            if PID_FILE.read_text().strip() == str(os.getpid()):
                PID_FILE.unlink()
        except OSError:
            pass
        # cyclonedds listener / multiprocessing children can keep the interpreter alive after the
        # script's own cleanup (go-home etc. already ran): leave hard.
        import multiprocessing
        for c in multiprocessing.active_children():      # Dex3 control process (daemon child)
            c.terminate()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
