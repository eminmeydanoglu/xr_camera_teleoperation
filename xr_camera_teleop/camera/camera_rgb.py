"""Single RGB camera capture (ZED, Omron FH overhead, OpenCV). ZED UVC fallback never computes stereo depth."""
from pathlib import Path
import cv2


def left_uvc_rgb(frame):
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[1] < 2*frame.shape[0] or frame.shape[1] % 2:
        raise RuntimeError("ZED UVC did not provide an expected side-by-side RGB frame")
    return frame[:, :frame.shape[1]//2].copy()


def zed_uvc_device():
    for name in sorted(Path("/sys/class/video4linux").glob("video*/name")):
        if "zed" in name.read_text().lower():
            return "/dev/" + name.parent.name
    raise RuntimeError("ZED is not connected: no ZED RGB/UVC device found (laptop camera was not selected)")


def open_rgb_camera(source):
    if source == "zed":
        try:
            import pyzed.sl as sl
        except (ImportError, OSError):
            sl = None
        if sl is not None:
            camera = sl.Camera()
            init = sl.InitParameters()
            init.camera_resolution = sl.RESOLUTION.HD720
            init.camera_fps = 30
            init.depth_mode = sl.DEPTH_MODE.NONE
            if camera.open(init) != sl.ERROR_CODE.SUCCESS:
                camera.close()
                raise RuntimeError("ZED RGB camera could not be opened")
            image = sl.Mat()
            info = camera.get_camera_information()
            calibration = info.camera_configuration.calibration_parameters.left_cam
            resolution = info.camera_configuration.resolution
            intrinsics = {"fx": float(calibration.fx), "fy": float(calibration.fy),
                "cx": float(calibration.cx), "cy": float(calibration.cy),
                "width": int(resolution.width), "height": int(resolution.height),
                "serial": str(info.serial_number), "side": "left", "distortion": [0.] * 5,
                "source": "zed-sdk-rectified-left-calibration"}
            def capture():
                if camera.grab() == sl.ERROR_CODE.SUCCESS:
                    camera.retrieve_image(image, sl.VIEW.LEFT)
                    return image.get_data()[:, :, :3].copy()
                return None
            return capture, camera.close, {"backend": "zed-sdk-left-rgb", "depth": False,
                                          "rectified": True, "intrinsics": intrinsics}
        from camera_intrinsics import discover_zed_serial, factory_intrinsics
        serial = discover_zed_serial()
        camera = cv2.VideoCapture(zed_uvc_device(), cv2.CAP_V4L2)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        camera.set(cv2.CAP_PROP_FPS, 30)
        if not camera.isOpened():
            camera.release()
            raise RuntimeError("ZED RGB/UVC device could not be opened")
        try:
            intrinsics = factory_intrinsics(serial, int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))//2,
                int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                Path(__file__).resolve().parents[2] / "data/camera")
        except Exception:
            camera.release()
            raise
        def capture():
            ok, frame = camera.read()
            return left_uvc_rgb(frame) if ok else None
        return capture, camera.release, {"backend": "zed-uvc-left-rgb", "depth": False,
            "rectified": False, "raw_intrinsics": intrinsics}
    if source == "omron":
        from camera_omron import OmronCamera, load_calibration
        camera = OmronCamera()
        try:
            intrinsics = load_calibration(*camera.size)
        except Exception:
            camera.close()
            raise
        return camera.read, camera.close, {"backend": "omron-fh-overhead", "depth": False,
            "rectified": False, "raw_intrinsics": intrinsics}
    camera = cv2.VideoCapture(int(source) if source.isdigit() else source)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, 30)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError("RGB source could not be opened: " + source)
    def capture():
        ok, frame = camera.read()
        return frame if ok else None
    return capture, camera.release, {"backend": "opencv-rgb", "depth": False, "rectified": False}
