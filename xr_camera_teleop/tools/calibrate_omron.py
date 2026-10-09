"""Checkerboard calibration of the overhead Omron FH camera -> configs/camera/omron-fh.json.

Live:     data/wilor-venv/bin/python xr_camera_teleop/tools/calibrate_omron.py
Offline:  ... calibrate_omron.py --images data/camera/omron-calib --table-distance-cm 100

Start with the board lying flat on the table (that view also measures camera height and tilt),
then hold it at different places and tilts (30-45 deg, all four directions, near and far).
Views are captured automatically when the board is found in a new enough pose.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "xr_camera_teleop" / "camera"))
from camera_omron import CALIBRATION, OmronCamera  # noqa: E402

FLAGS = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY


def detect(gray, pattern):
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=FLAGS)
    return corners.reshape(-1, 2) if ok else None


def novel(corners, views, min_shift):
    """New when the corners moved enough vs every captured view (position, scale or tilt)."""
    return all(np.abs(corners - v).mean() > min_shift for v in views)


def board_points(pattern, square_m):
    cols, rows = pattern
    grid = np.zeros((rows * cols, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return grid


def calibrate(views, size, pattern, square_m, focal=None):
    obj = [board_points(pattern, square_m)] * len(views)
    img = [v.reshape(-1, 1, 2).astype(np.float32) for v in views]
    flags = cv2.CALIB_FIX_K3
    k0 = None
    if focal is not None:   # focal fixed (fx = fy); principal point and distortion still optimised
        k0 = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1.0]])
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH
    rms, k, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, size, k0, None, flags=flags)
    per_view = []
    for o, i, r, t in zip(obj, img, rvecs, tvecs):
        p, _ = cv2.projectPoints(o, r, t, k, dist)
        per_view.append(float(np.sqrt(np.mean(np.sum((p - i) ** 2, axis=2)))))
    return rms, k, dist.ravel(), rvecs, tvecs, per_view


def table_pose(rvec, tvec, pattern, square_m):
    """Distance from the camera to the board plane and the optical axis tilt from its normal."""
    R, _ = cv2.Rodrigues(rvec)
    normal = R[:, 2]                                 # board z axis in camera frame
    distance = abs(float(normal @ tvec.ravel()))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))
    return distance, tilt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--cols", type=int, default=12, help="inner corners per row")
    ap.add_argument("--rows", type=int, default=17, help="inner corners per column")
    ap.add_argument("--square-mm", type=float, default=20.0)
    ap.add_argument("--views", type=int, default=25, help="views to capture")
    ap.add_argument("--min-shift", type=float, default=25.0, help="px mean corner shift for a new view")
    ap.add_argument("--images", type=Path, default=None, help="calibrate from saved images instead of live")
    ap.add_argument("--save-dir", type=Path, default=ROOT / "data/camera/omron-calib")
    ap.add_argument("--out", type=Path, default=CALIBRATION)
    ap.add_argument("--table-distance-cm", type=float, default=None,
                    help="tape-measured camera-to-table distance; fixes the focal length via the flat first view")
    ap.add_argument("--board-thickness-mm", type=float, default=3.0)
    ap.add_argument("--show", action=argparse.BooleanOptionalAction, default=True, help="preview window")
    a = ap.parse_args()
    pattern, square_m = (a.cols, a.rows), a.square_mm / 1000.0

    views, names, size = [], [], None
    if a.images:
        for path in sorted(a.images.glob("*.png")):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            size = gray.shape[::-1]
            c = detect(gray, pattern)
            print(path.name, "ok" if c is not None else "board not found")
            if c is not None:
                views.append(c)
                names.append(path.name)
    else:
        cam = OmronCamera()
        size = cam.size
        a.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print(f"Capturing {a.views} views at {size[0]}x{size[1]}; board {a.cols}x{a.rows} inner corners, "
              f"{a.square_mm:g} mm. Start flat on the table, then tilt/move it. q = finish early.", flush=True)
        last_print = 0.0
        try:
            while len(views) < a.views:
                frame = cam.read()
                if frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                c = detect(gray, pattern)
                status = "board not found"
                if c is not None:
                    if novel(c, views, a.min_shift):
                        name = f"{stamp}-{len(views):02d}.png"
                        cv2.imwrite(str(a.save_dir / name), frame)
                        views.append(c)
                        names.append(name)
                        status = f"captured view {len(views)}/{a.views}"
                        print(status, flush=True)
                    else:
                        status = "found - move/tilt the board to a new pose"
                elif time.monotonic() - last_print > 3:
                    print(status, flush=True)
                    last_print = time.monotonic()
                if a.show:
                    view = frame.copy()
                    if c is not None:
                        cv2.drawChessboardCorners(view, pattern, c.reshape(-1, 1, 2), True)
                    cv2.putText(view, f"{len(views)}/{a.views}  {status}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Omron calibration", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cam.close()
            if a.show:
                cv2.destroyAllWindows()

    if len(views) < 8:
        sys.exit(f"Only {len(views)} usable views; need at least 8 (ideally 20+ with tilted poses).")
    rms, k, dist, rvecs, tvecs, per_view = calibrate(views, size, pattern, square_m)
    free_fx = float(k[0, 0])
    if a.table_distance_cm is not None:
        # Nearly fronto-parallel views barely constrain the focal length (the reprojection error is
        # flat over +-10%), but the first view lies flat on the table: pick the focal length that
        # puts that board at the measured distance. Distance scales linearly with the focal length.
        target = a.table_distance_cm / 100.0 - a.board_thickness_mm / 1000.0
        focal = free_fx
        for _ in range(6):
            rms, k, dist, rvecs, tvecs, per_view = calibrate(views, size, pattern, square_m, focal)
            distance, _ = table_pose(rvecs[0], tvecs[0], pattern, square_m)
            focal *= target / distance
        rms, k, dist, rvecs, tvecs, per_view = calibrate(views, size, pattern, square_m, focal)
    fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
    distance, tilt = table_pose(rvecs[0], tvecs[0], pattern, square_m)
    result = {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
              "width": int(size[0]), "height": int(size[1]),
              "distortion": [float(v) for v in dist[:5]], "distortion_model": "opencv-brown5",
              "rms_px": float(rms), "views": len(views), "per_view_rms_px": dict(zip(names, per_view)),
              "pattern_inner_corners": list(pattern), "square_mm": a.square_mm,
              "hfov_deg": float(np.degrees(2 * np.arctan(size[0] / (2 * fx)))),
              "first_view_board_distance_m": distance, "first_view_optical_axis_tilt_deg": tilt,
              "focal_constraint": None if a.table_distance_cm is None else {
                  "table_distance_cm": a.table_distance_cm, "board_thickness_mm": a.board_thickness_mm,
                  "unconstrained_fx": free_fx},
              "date": datetime.datetime.now().isoformat(timespec="seconds"),
              "source": "omron-fh-checkerboard"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_view_rms_px"}, indent=2))
    print(f"worst views: {sorted(zip(per_view, names), reverse=True)[:3]}")
    print(f"first view (table): board {distance * 100:.1f} cm from camera, optical axis {tilt:.2f} deg off vertical")
    print("saved", a.out)


if __name__ == "__main__":
    main()
