"""Frame sources: ZED (SDK rectified preferred, UVC + factory undistort fallback),
Omron FH overhead camera ("omron"), OpenCV camera index, or video file. Each source yields BGR uint8 frames and
carries pinhole intrinsics for the frames it returns."""
import math
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xr_camera_teleop" / "camera"))
sys.path.insert(0, str(ROOT))


def assumed_intrinsics(w, h, hfov_deg=70.0):
    fx = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    return {"fx": fx, "fy": fx, "cx": w / 2, "cy": h / 2, "width": w, "height": h,
            "source": f"assumed-hfov{hfov_deg:g}deg (uncalibrated)"}


class Source:
    info = {}
    intrinsics = None
    paced = False   # True for video files (we pace to file fps)

    def read(self):
        raise NotImplementedError

    def level(self):
        """3x3 camera levelling rotation from gravity (CONTRACT.md "cam_R_level") or None."""
        return None

    def level_mean(self):
        """Mean levelling over the first second of frames (for the recording sidecar) or None."""
        return None

    def close(self):
        pass


class ZedSdkSource(Source):
    def __init__(self):
        import pyzed.sl as sl
        self.sl = sl
        self.cam = sl.Camera()
        p = sl.InitParameters()
        p.camera_resolution = sl.RESOLUTION.HD720
        p.camera_fps = 30
        p.depth_mode = sl.DEPTH_MODE.NONE
        err = self.cam.open(p)
        if err != sl.ERROR_CODE.SUCCESS:
            self.cam.close()
            raise RuntimeError(f"ZED SDK open failed: {err}")
        self.mat = sl.Mat()
        self.sens = sl.SensorsData()
        self._down = None          # low-passed gravity ("down") direction in the left-camera frame
        self._down_first = []      # first ~30 samples -> sidecar mean
        info = self.cam.get_camera_information()
        c = info.camera_configuration.calibration_parameters.left_cam
        r = info.camera_configuration.resolution
        self.intrinsics = {"fx": float(c.fx), "fy": float(c.fy), "cx": float(c.cx),
                           "cy": float(c.cy), "width": int(r.width), "height": int(r.height),
                           "source": "zed-sdk-rectified-left"}
        self.info = {"backend": "zed-sdk", "rectified": True, "serial": str(info.serial_number)}

    def read(self):
        sl = self.sl
        if self.cam.grab() != sl.ERROR_CODE.SUCCESS:
            return None
        self.cam.retrieve_image(self.mat, sl.VIEW.LEFT)
        self._update_gravity()
        return np.ascontiguousarray(self.mat.get_data()[:, :, :3])

    def _update_gravity(self):
        """Accelerometer (specific force, points UP at rest) -> down = -a/|a|; EMA-filtered.
        The SDK reports IMU data in the camera frame for the default IMAGE coordinate system.
        (Untested without a ZED 2i on this host.)"""
        try:
            sl = self.sl
            if self.cam.get_sensors_data(self.sens, sl.TIME_REFERENCE.IMAGE) != sl.ERROR_CODE.SUCCESS:
                return
            a = np.asarray(self.sens.get_imu_data().get_linear_acceleration(), float)
            n = np.linalg.norm(a)
            if not np.isfinite(n) or not 5.0 < n < 15.0:      # not ~1 g: ignore (shaking / invalid)
                return
            d = -a / n
            self._down = d if self._down is None else 0.9 * self._down + 0.1 * d
            self._down /= np.linalg.norm(self._down)
            if len(self._down_first) < 30:
                self._down_first.append(self._down.copy())
        except Exception:
            pass

    def level(self):
        from xr_camera_teleop.teleop.camlevel import level_from_down
        return None if self._down is None else level_from_down(self._down).tolist()

    def level_mean(self):
        from xr_camera_teleop.teleop.camlevel import level_from_down
        if not self._down_first:
            return None
        return level_from_down(np.mean(self._down_first, axis=0)).tolist()

    def close(self):
        self.cam.close()


class ZedUvcSource(Source):
    """Raw side-by-side YUYV 2560x720 -> left half -> factory undistortion (same K)."""

    def __init__(self):
        from camera_intrinsics import MonoUndistorter, discover_zed_serial, factory_intrinsics
        from camera_rgb import zed_uvc_device
        dev = zed_uvc_device()
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"ZED UVC {dev} could not be opened (busy?)")
        self.cap = cap
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw = factory_intrinsics(discover_zed_serial(), w, h, ROOT / "data/camera")
        self.und = MonoUndistorter(raw)
        self.intrinsics = dict(self.und.intrinsics)
        self.info = {"backend": "zed-uvc", "rectified": False, "device": dev,
                     "undistorted": "factory-brown5-mono"}

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            return None
        if f.ndim != 3 or f.shape[1] != 2 * self.und.size[0]:
            raise RuntimeError(f"unexpected UVC frame shape {f.shape}")
        return self.und.apply(np.ascontiguousarray(f[:, :f.shape[1] // 2]))

    def close(self):
        self.cap.release()


class OmronSource(Source):
    """Fixed overhead Omron FH camera (Raider bridge /dev/omron-fh) looking straight down.

    Checkerboard intrinsics + mono undistortion (same K). The camera has no IMU but never
    moves, so cam_R_level is the constant straight-down levelling (image-up = forward)."""

    def __init__(self):
        from camera_intrinsics import MonoUndistorter
        from camera_omron import LEVEL_STRAIGHT_DOWN, OmronCamera, load_calibration
        self.cam = OmronCamera()
        try:
            calib = load_calibration(*self.cam.size)
        except Exception:
            self.cam.close()
            raise
        keep = ("fx", "fy", "cx", "cy", "width", "height", "distortion", "distortion_model", "source")
        self.und = MonoUndistorter({k: calib[k] for k in keep})
        self.intrinsics = dict(self.und.intrinsics)
        self._level = LEVEL_STRAIGHT_DOWN
        # overhead: every hand above the table is the operator's; drop only implausible depths
        table = (calib.get("focal_constraint") or {}).get("table_distance_cm", 100.0) / 100.0
        self.max_hand_depth = round(table + 0.05, 3)
        self.info = {"backend": "omron-fh", "rectified": False, "region": list(self.cam.region),
                     "undistorted": "checkerboard-brown5-mono", "calibration": calib["calibration_file"],
                     "calibration_rms_px": calib.get("rms_px")}

    def read(self):
        frame = self.cam.read()
        return None if frame is None else self.und.apply(frame)

    def level(self):
        return self._level

    def level_mean(self):
        return self._level

    def close(self):
        self.cam.close()


class CvSource(Source):
    def __init__(self, spec):
        is_cam = spec.isdigit()
        self.cap = cv2.VideoCapture(int(spec) if is_cam else spec)
        if is_cam:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("source could not be opened: " + spec)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.intrinsics = assumed_intrinsics(w, h)
        sidecar = Path(spec).with_suffix(".json") if not is_cam else None
        if sidecar is not None and sidecar.exists():  # written by server.py --record-video
            self.intrinsics = json.loads(sidecar.read_text())["intrinsics"]
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.paced = not is_cam
        self.period = 1.0 / fps if 1 < fps < 240 else 1 / 30
        self.info = {"backend": "opencv-" + ("camera" if is_cam else "file"), "rectified": False}
        self.next_t = time.monotonic()

    def read(self):
        if self.paced:
            self.next_t += self.period
            time.sleep(max(0.0, self.next_t - time.monotonic()))
        ok, f = self.cap.read()
        if not ok and self.paced:  # loop files
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = self.cap.read()
        return f if ok else None

    def close(self):
        self.cap.release()


class ImageSource(CvSource):
    """Single image repeated at a fixed rate (verification without a camera)."""

    def __init__(self, path, fps=30.0):
        self.img = cv2.imread(path)
        if self.img is None:
            raise RuntimeError("cannot read image " + path)
        h, w = self.img.shape[:2]
        self.intrinsics = assumed_intrinsics(w, h)
        self.paced, self.period, self.next_t = True, 1 / fps, time.monotonic()
        self.info = {"backend": "image", "rectified": False}

    def read(self):
        self.next_t += self.period
        time.sleep(max(0.0, self.next_t - time.monotonic()))
        return self.img.copy()

    def close(self):
        pass


def open_source(spec):
    if spec == "zed":
        try:
            s = ZedSdkSource()
            print("[source] ZED SDK (rectified left) opened", flush=True)
            return s
        except Exception as exc:  # no SDK, no CUDA (needs driver), camera busy ...
            print(f"[source] ZED SDK unavailable ({exc!r}); falling back to UVC", flush=True)
        return ZedUvcSource()
    if spec == "omron":
        s = OmronSource()
        print("[source] Omron FH overhead camera opened", s.info, flush=True)
        return s
    if Path(spec).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        return ImageSource(spec)
    return CvSource(spec)
