"""Omron FH overhead camera via the Raider camera bridge (/dev/omron-fh).

The bridge (~/camera-bridge, Windows omron-sender.exe -> receiver.py) writes 1280x720 YUYV with
the full sensor field of view scaled to fit and black side bars. This module crops the bars,
returns only fresh frames (None while the bridge reports no signal) and loads the checkerboard
calibration from configs/camera/omron-fh.json (tools: xr_camera_teleop/tools/calibrate_omron.py).

The camera is fixed and looks straight down at the table; the operator stands at the bottom
edge of the image, so image-up is the operator's forward direction.
"""
import json
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

DEVICE = "/dev/omron-fh"
STATUS_URL = "http://127.0.0.1:8766/status"
OUT_W, OUT_H = 1280, 720                 # receiver.py output size
DEFAULT_SOURCE = (1224, 1024)            # sensor 2448x2048 binned 2x2 (sender mode=full)
CALIBRATION = Path(__file__).resolve().parents[2] / "configs/camera/omron-fh.json"
# v_level = R @ v_cam: optical axis (+z_cam) is gravity (+y_level), image-up (-y_cam) is forward.
# Same as camlevel.pitch_rotation(90) and level_from_down([0, 0, 1]).
LEVEL_STRAIGHT_DOWN = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]


def bridge_status(timeout=0.5):
    with urllib.request.urlopen(STATUS_URL, timeout=timeout) as response:
        return json.loads(response.read())


def active_region(source_size):
    """(x, y, w, h) of the camera image inside the 1280x720 frame; mirrors receiver.fit_frame."""
    w, h = source_size
    scale = min(OUT_W / w, OUT_H / h)
    nw, nh = int(round(w * scale)) & ~1, int(round(h * scale)) & ~1
    return (OUT_W - nw) // 2, (OUT_H - nh) // 2, nw, nh


def load_calibration(width, height, path=CALIBRATION):
    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"Omron camera is not calibrated: {path} missing. "
                           "Run xr_camera_teleop/tools/calibrate_omron.py with the checkerboard.")
    calib = json.loads(path.read_text())
    if (calib["width"], calib["height"]) != (width, height):
        raise RuntimeError(f"Omron calibration is for {calib['width']}x{calib['height']} but the bridge "
                           f"delivers {width}x{height} (sender mode changed?); recalibrate.")
    calib["calibration_file"] = str(path)
    return calib


class OmronCamera:
    def __init__(self, device=DEVICE):
        try:
            status = bridge_status()
        except OSError as exc:
            raise RuntimeError(f"Omron camera bridge is not running ({STATUS_URL}: {exc})") from exc
        res = status.get("source_resolution")
        size = tuple(int(v) for v in res.split("x")) if res else DEFAULT_SOURCE
        self.region = active_region(size)
        self.stale = bool(status.get("stale", True))
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Omron camera {device} could not be opened")
        fw, fh = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (fw, fh) != (OUT_W, OUT_H):
            self.cap.release()
            raise RuntimeError(f"Omron camera delivers {fw}x{fh}, expected {OUT_W}x{OUT_H}")
        self.size = self.region[2], self.region[3]
        self._stop = threading.Event()
        threading.Thread(target=self._watch, daemon=True).start()

    def _watch(self):
        """Poll the bridge so frozen/no-signal frames are never returned as camera images."""
        while not self._stop.wait(0.25):
            try:
                self.stale = bool(bridge_status()["stale"])
            except (OSError, ValueError, KeyError):
                self.stale = True

    def read(self):
        ok, frame = self.cap.read()
        if not ok or self.stale:
            if not ok:
                time.sleep(0.01)
            return None
        x, y, w, h = self.region
        return np.ascontiguousarray(frame[y:y + h, x:x + w])

    def close(self):
        self._stop.set()
        self.cap.release()
