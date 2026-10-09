"""Glue to use third_party/xr_teleoperate as-is (no edits, imports only).

* load_tv_wrapper(): load televuer/tv_wrapper.py WITHOUT importing the vuer
  stack (TeleVuer is stubbed), so we reuse the real TeleData dataclass and the
  real frame constants (T_TO_UNITREE_HAND, T_ROBOT_OPENXR, CONST_HEAD_POSE...).
* make_arm_ik() / make_hand_retargeting(): G1_29_ArmIK (self-collision-aware subclass by default,
  see safe_arm_ik.py) and HandRetargeting use
  cwd-relative paths ('../assets/...', cache pickle in cwd). We chdir to
  <repo>/data/xr_run/teleop (which has ../assets -> third_party assets symlink)
  while constructing them, so nothing is written under third_party/.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
XRT = REPO / "third_party" / "xr_teleoperate"
RUN_DIR = REPO / "data" / "xr_run" / "teleop"
ASSETS = XRT / "assets"
G1_MJCF = ASSETS / "g1" / "g1_body29_hand14.xml"


def _ensure_run_dir():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    link = RUN_DIR.parent / "assets"
    if not link.exists():
        link.symlink_to(os.path.relpath(ASSETS, RUN_DIR.parent))


def add_xrt_to_path():
    p = str(XRT)
    if p not in sys.path:
        sys.path.insert(0, p)


_TV = None


def load_tv_wrapper():
    """Return the real tv_wrapper module (TeleData, constants), vuer stubbed."""
    global _TV
    if _TV is not None:
        return _TV
    src_dir = XRT / "teleop" / "televuer" / "src" / "televuer"
    pkg = types.ModuleType("_xrt_tv")
    pkg.__path__ = [str(src_dir)]
    sys.modules["_xrt_tv"] = pkg
    stub = types.ModuleType("_xrt_tv.televuer")
    stub.TeleVuer = object
    sys.modules["_xrt_tv.televuer"] = stub
    spec = importlib.util.spec_from_file_location("_xrt_tv.tv_wrapper", src_dir / "tv_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_xrt_tv.tv_wrapper"] = mod
    spec.loader.exec_module(mod)
    _TV = mod
    return mod


class _chdir:
    def __init__(self, d):
        self.d = str(d)

    def __enter__(self):
        self.old = os.getcwd()
        os.chdir(self.d)

    def __exit__(self, *a):
        os.chdir(self.old)


def make_arm_ik(self_collision=None, **kw):
    """G1_29_ArmIK, by default the self-collision-aware subclass (teleop/safe_arm_ik.py).

    self_collision=None -> env WILOR_XR_SELF_COLLISION (default on; 0/false/off = upstream IK).
    kw (margin=, constraints=, gate=) go to SafeG1_29_ArmIK."""
    _ensure_run_dir()
    add_xrt_to_path()
    with _chdir(RUN_DIR):
        from xr_camera_teleop.teleop.safe_arm_ik import SafeG1_29_ArmIK, UpstreamG1_29_ArmIK, self_collision_enabled_by_env
        if self_collision is None:
            self_collision = self_collision_enabled_by_env()
        if self_collision:
            return SafeG1_29_ArmIK(**kw)
        return UpstreamG1_29_ArmIK()


def connect_ik(ik, wrapper, clock=None):
    """Wire a SafeG1_29_ArmIK to a WiLoRTeleWrapper: wrapper.reset() / handedness swap -> joint-space
    return to home (ik.reset_to_home); optional shared (virtual) clock for the IK's rate limits.
    No-op for the upstream IK."""
    if hasattr(ik, "reset_to_home"):
        wrapper.on_home_request = ik.reset_to_home
        if clock is not None:
            ik.clock = clock


def make_hand_retargeting():
    _ensure_run_dir()
    add_xrt_to_path()
    with _chdir(RUN_DIR):
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
        return HandRetargeting(HandType.UNITREE_DEX3)
