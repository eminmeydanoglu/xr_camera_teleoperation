"""Factory RGB intrinsics and mono undistortion; no stereo/depth computation."""
import configparser
import hashlib
import math
from pathlib import Path
import urllib.request

import cv2
import numpy as np

RESOLUTIONS = {(1280, 720): "HD", (1920, 1080): "FHD",
               (2208, 1242): "2K", (672, 376): "VGA"}


def discover_zed_serial(sys_root=Path("/sys/bus/usb/devices")):
    serials = set()
    for device in sys_root.iterdir():
        try:
            if (device / "idVendor").read_text().strip() != "2b03":
                continue
            # ZED 2i UVC reports generic OV0001. The HID device contains the
            # actual camera serial used by the vendor's calibration service.
            if (device / "idProduct").read_text().strip() == "f881":
                value = (device / "serial").read_text().strip()
                if value.isdigit():
                    serials.add(value)
        except OSError:
            continue
    if len(serials) != 1:
        raise RuntimeError(f"Expected one unambiguous ZED 2i serial, found {sorted(serials)}")
    return next(iter(serials))


def validate_intrinsics(intrinsics, width=None, height=None):
    for key in ("fx", "fy", "cx", "cy"):
        if not math.isfinite(float(intrinsics[key])):
            raise ValueError("Non-finite camera intrinsic: " + key)
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0:
        raise ValueError("Focal lengths must be positive pixel values")
    if width is not None and (intrinsics["width"], intrinsics["height"]) != (width, height):
        raise ValueError("Camera intrinsics resolution does not match the inference image")
    return intrinsics


def parse_factory_calibration(text, serial, width, height):
    if (width, height) not in RESOLUTIONS:
        raise ValueError(f"No explicit ZED factory mode for {width}x{height}")
    config = configparser.ConfigParser()
    config.read_string(text)
    section = "LEFT_CAM_" + RESOLUTIONS[(width, height)]
    values = config[section]
    result = {key: float(values[key]) for key in ("fx", "fy", "cx", "cy")}
    # Match Stereolabs zed-open-capture calibration.hpp's Brown-Conrady
    # LEFT_CAM_<mode> coefficients, in OpenCV's documented order.
    result.update(width=width, height=height, serial=str(serial), side="left",
                  distortion=[float(values[k]) for k in ("k1", "k2", "p1", "p2", "k3")],
                  distortion_model="opencv-brown5", factory_section=section,
                  source=f"https://calib.stereolabs.com/?SN={serial}")
    validate_intrinsics(result, width, height)
    if not all(math.isfinite(v) for v in result["distortion"]):
        raise ValueError("Invalid factory distortion coefficients")
    return result


def factory_intrinsics(serial, width, height, cache_dir):
    path = Path(cache_dir) / f"SN{serial}.conf"
    if path.exists():
        raw = path.read_bytes()
    else:
        with urllib.request.urlopen(f"https://calib.stereolabs.com/?SN={serial}", timeout=15) as response:
            raw = response.read()
        parse_factory_calibration(raw.decode(), serial, width, height)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".conf.tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)
    result = parse_factory_calibration(raw.decode(), serial, width, height)
    result["calibration_sha256"] = hashlib.sha256(raw).hexdigest()
    result["calibration_file"] = str(path)
    return result


class MonoUndistorter:
    """Single left-camera image, identity rotation and unchanged output K."""
    def __init__(self, intrinsics):
        validate_intrinsics(intrinsics)
        self.raw_intrinsics = dict(intrinsics)
        self.intrinsics = {**intrinsics, "distortion": [0.] * 5,
                           "source": intrinsics["source"] + "; mono-undistorted-same-K"}
        k = np.array([[intrinsics["fx"], 0., intrinsics["cx"]],
                      [0., intrinsics["fy"], intrinsics["cy"]], [0., 0., 1.]])
        self.size = intrinsics["width"], intrinsics["height"]
        self.map_x, self.map_y = cv2.initUndistortRectifyMap(k,
            np.asarray(intrinsics["distortion"], dtype=np.float64), np.eye(3), k,
            self.size, cv2.CV_32FC1)

    def apply(self, image):
        if (image.shape[1], image.shape[0]) != self.size:
            raise ValueError("Captured RGB resolution changed; intrinsics must be reloaded")
        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR)


def calibrated_crop_translation(cam, box_center, box_size, intrinsics):
    """WiLoR weak-perspective crop -> actual calibrated full-camera translation.

    Keep native MANO geometry/rotation and learned crop offsets unchanged.
    Horizontal weak scale anchors Z using fx; center offsets use fx and fy.
    """
    validate_intrinsics(intrinsics)
    cam = np.asarray(cam, dtype=np.float64)
    center = np.asarray(box_center, dtype=np.float64)
    size = np.asarray(box_size, dtype=np.float64).reshape(-1)
    if cam.shape != (len(size), 3) or center.shape != (len(size), 2):
        raise ValueError("Inconsistent crop camera/center/size shapes")
    denominator = size * cam[:, 0]
    if not np.isfinite(cam).all() or not np.isfinite(center).all() or not np.isfinite(denominator).all() or np.any(denominator <= 1e-8):
        raise ValueError("Invalid weak-perspective scale or camera prediction")
    z = 2 * intrinsics["fx"] / denominator
    x = cam[:, 1] + (center[:, 0] - intrinsics["cx"]) * z / intrinsics["fx"]
    y = cam[:, 2] + (center[:, 1] - intrinsics["cy"]) * z / intrinsics["fy"]
    return np.column_stack((x, y, z)).astype(np.float32)


def project_camera_points(points, intrinsics):
    validate_intrinsics(intrinsics)
    points = np.asarray(points)
    return np.column_stack((points[:, 0] / points[:, 2] * intrinsics["fx"] + intrinsics["cx"],
                            points[:, 1] / points[:, 2] * intrinsics["fy"] + intrinsics["cy"]))
