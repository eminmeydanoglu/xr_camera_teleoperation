"""MuJoCo floating-hands viewer driven by the WiLoR stream.

  python -m xr_camera_teleop.viz.floating_hands --addr tcp://127.0.0.1:5590 --mode relative
Keys (viewer): C calibrate (relative) / clear toggles off not needed, F toggle smoothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from xr_camera_teleop.viz.mapping import (HANDS, WAIST_DROP, Extrapolator, HandMapper,  # noqa: E402
                                  HandSmoother, robot_anchors)
from xr_camera_teleop.teleop.camlevel import level_rotation  # noqa: E402
from xr_camera_teleop.teleop.clutch import StillDetector  # noqa: E402
from xr_camera_teleop.viz.stream import make_client  # noqa: E402

HIDE_AFTER = 2.0  # seconds a hand may be missing before it vanishes (grey until then)


class HandsApp:
    def __init__(self, addr, mode, scale, cam_height, smooth=False, neutral="viewer",
                 waist_height=None, extrap=True, cam_pitch_deg=0.0, auto_calibrate=True):
        from xr_camera_teleop.viz.hands_scene import HandsScene

        self.waist_height = cam_height - WAIST_DROP if waist_height is None else waist_height
        self.scene = HandsScene(cam_height, waist_height=self.waist_height if neutral == "robot" else None)
        self.client = make_client(addr)
        if neutral == "robot":  # same neutral wrist poses (R=I) the clutch starts from
            self.mapper = HandMapper(mode, scale, cam_height, neutral_rot="ref",
                                     anchors=robot_anchors(self.waist_height), max_depth=0.9)
        else:
            self.mapper = HandMapper(mode, scale, cam_height, max_depth=0.9 if mode == "relative" else None)
        self.extrap = {h: Extrapolator() for h in HANDS} if extrap else None
        self.smooth = smooth
        self.smoothers = {h: HandSmoother() for h in HANDS}
        self.last_seen = {h: -1e9 for h in HANDS}
        self.last_cam = {}  # side -> (pos, rot, kps) latest raw (smoothed) in CAM
        self.want_calibrate = False
        self.last_frame_id = None
        self.cam_pitch_deg = cam_pitch_deg
        self.auto_calibrate = auto_calibrate
        self.still = StillDetector(HANDS, 1.0, 0.03)

    # key handlers (called from viewer thread; only set flags)
    def key(self, keycode):
        k = chr(keycode).upper() if 0 < keycode < 256 else ""
        if k == "C":
            self.want_calibrate = True
        elif k == "F":
            self.smooth = not self.smooth
            for s in self.smoothers.values():
                s.reset()
            print("smoothing", "ON" if self.smooth else "OFF")

    def step(self):
        now = time.monotonic()
        msg = self.client.latest(max_age=0.3)
        fresh = {}
        if msg is not None and msg.get("frame_id") != self.last_frame_id:
            self.last_frame_id = msg.get("frame_id")
            self.mapper.R_level = level_rotation(msg, self.cam_pitch_deg)
            if self.mapper.max_depth and "max_hand_depth" in msg:
                self.mapper.max_depth = msg["max_hand_depth"]
            for h in HANDS:
                if h not in msg.get("hands", {}):
                    self.still.lost(h)
            max_depth = msg.get("max_hand_depth", self.mapper.max_depth) if self.mapper.max_depth else None
            for h, d in msg.get("hands", {}).items():
                if h not in HANDS or (max_depth and d["wrist_pos"][2] > max_depth):
                    continue
                p = np.asarray(d["wrist_pos"], float)
                R = np.asarray(d["wrist_rot"], float)
                k = np.asarray(d["keypoints"], float)
                if self.smooth:
                    p, R, k = self.smoothers[h](p, R, k, now)
                fresh[h] = (p, R, k)
                self.last_seen[h] = now
                self.last_cam[h] = (p, R, k)
                self.still.push(h, now, self.mapper.R_level @ p)
                if self.extrap:
                    self.extrap[h].update(msg.get("t_publish", now), p, R, k)
        if msg is None or (msg is not None and not msg.get("hands")):
            pass
        if (self.auto_calibrate and self.mapper.mode == "relative" and not self.mapper.calibrated
                and self.still.ready()):
            self.want_calibrate = True      # both hands nearly still for 1 s
        if self.want_calibrate:
            self.want_calibrate = False
            seen = {h: (v[0], v[1]) for h, v in self.last_cam.items()
                    if now - self.last_seen[h] < 0.3}
            if seen:
                self.mapper.calibrate(seen)
                print("calibrated:", sorted(seen))
        for h in HANDS:
            age = now - self.last_seen[h]
            if h not in self.last_cam or age > HIDE_AFTER:
                self.scene.set_state(h, "hidden")
                continue
            cam = self.extrap[h].at(now) if self.extrap and self.extrap[h].cur else self.last_cam[h]
            res = self.mapper.map(h, *cam)
            if res is None:  # relative, uncalibrated: show raw absolute as grey hint
                self.scene.set_state(h, "hidden")
                continue
            self.scene.set_state(h, "live" if age < 0.3 else "stale")
            self.scene.pose(h, *res)
        self.scene.forward()


def _camera(cam, cam_height):
    from xr_camera_teleop.viz.hands_scene import default_camera

    look, dist, az, el = default_camera(cam_height)
    cam.lookat[:] = look
    cam.distance, cam.azimuth, cam.elevation = dist, az, el


def run_live(app, cam_height):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(app.scene.model, app.scene.data,
                                      key_callback=app.key) as v:
        _camera(v.cam, cam_height)
        print("keys: C = calibrate (relative mode), F = toggle smoothing")
        period, nxt = 1 / 60, time.monotonic()
        while v.is_running():
            with v.lock():
                app.step()
            v.sync()
            nxt += period                      # absolute schedule: no drift, no catch-up burst
            now = time.monotonic()
            if nxt < now - period:
                nxt = now
            time.sleep(max(0.0, nxt - now))


def run_headless(app, cam_height, frames, snapshot, calibrate_at, size=(1280, 720)):
    import mujoco

    renderer = None
    for gl in (os.environ.get("MUJOCO_GL"), "egl", "osmesa", "glfw"):
        if not gl:
            continue
        os.environ["MUJOCO_GL"] = gl
        try:
            renderer = mujoco.Renderer(app.scene.model, size[1], size[0])
            print("renderer:", gl)
            break
        except Exception as e:  # noqa: BLE001
            print("GL backend", gl, "failed:", type(e).__name__)
    if renderer is None:
        raise SystemExit("no usable GL backend")
    cam = mujoco.MjvCamera()
    _camera(cam, cam_height)
    got = 0
    t_end = time.monotonic() + 20
    while got < frames and time.monotonic() < t_end:
        app.step()
        if app.last_frame_id is not None and app.last_cam:
            got += 1
            if got == calibrate_at and app.mapper.mode == "relative":
                app.want_calibrate = True
        time.sleep(1 / 30)
    app.step()
    renderer.update_scene(app.scene.data, cam)
    img = renderer.render()
    from PIL import Image

    Image.fromarray(img).save(snapshot)
    print("saved", snapshot, "frames:", got)
    sys.stdout.flush()
    os._exit(0)  # avoid noisy EGL teardown


def run_montage(app, cam_height, path, size=(640, 480)):
    """Offline: render synthetic hands at several curl phases side by side (no socket)."""
    import mujoco
    from PIL import Image

    from xr_camera_teleop.viz.synth import make_message

    os.environ.setdefault("MUJOCO_GL", "egl")
    r = mujoco.Renderer(app.scene.model, size[1], size[0])
    cam = mujoco.MjvCamera()
    _camera(cam, cam_height)
    tiles = []
    for t in (3.0, 3.5, 0.0, 0.5, 1.0):  # curl of left hand: 0 .. 1
        class _C:
            def latest(self, max_age=0.3, m=make_message(1, t)):
                return m
        app.client, app.last_frame_id = _C(), None
        app.step()
        r.update_scene(app.scene.data, cam)
        tiles.append(r.render().copy())
    Image.fromarray(np.concatenate(tiles, axis=1)).save(path)
    print("saved", path)
    sys.stdout.flush()
    os._exit(0)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="tcp://127.0.0.1:5590")
    ap.add_argument("--mode", choices=["absolute", "relative"], default="absolute")
    ap.add_argument("--scale", type=float, default=None,
                    help="absolute: position scale about the camera (default 1); relative: gain k (default 0.8)")
    ap.add_argument("--cam-height", type=float, default=1.5)
    ap.add_argument("--neutral", choices=["viewer", "robot"], default="viewer",
                    help="relative mode anchors: 'robot' = G1 neutral wrist poses in the waist frame "
                         "(what the robot receives); waist reference drawn")
    ap.add_argument("--waist-height", type=float, default=None,
                    help="waist reference height in W (default cam-height - 0.5)")
    ap.add_argument("--cam-pitch-deg", type=float, default=0.0,
                    help="camera pitch, positive = looking down (used when messages carry no cam_R_level)")
    ap.add_argument("--no-auto-calibrate", action="store_true",
                    help="relative mode: do not auto-calibrate when both hands are still for 1 s (keys: C)")
    ap.add_argument("--no-extrap", action="store_true",
                    help="disable 30 Hz -> 60 Hz constant-velocity extrapolation")
    ap.add_argument("--smooth", action="store_true", help="start with One-Euro smoothing on")
    ap.add_argument("--curl-montage", default=None, help="render synthetic curl phases to this PNG and exit")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--snapshot", default="snapshot.png")
    ap.add_argument("--calibrate-at", type=int, default=10,
                    help="headless relative mode: calibrate after this many frames")
    a = ap.parse_args(argv)
    app = HandsApp(a.addr, a.mode, a.scale, a.cam_height, a.smooth, a.neutral, a.waist_height,
                  not a.no_extrap, a.cam_pitch_deg, not a.no_auto_calibrate)
    if a.curl_montage:
        run_montage(app, a.cam_height, a.curl_montage)
    elif a.headless:
        run_headless(app, a.cam_height, a.frames, a.snapshot, a.calibrate_at)
    else:
        run_live(app, a.cam_height)


if __name__ == "__main__":
    main()
