"""Device-agnostic WiLoR engine producing contract-conformant hand dicts.

Reuses xr_camera_teleop/camera: camera_intrinsics (calibrated crop translation)
and preprocessing (OpenCV-equivalent Gaussian). Unlike the original wilor_teleop.inference
this runs on CPU too (slow; for verification only).
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TELEOP = ROOT / "xr_camera_teleop" / "camera"
# WiLoR model code + weights are NOT part of this repo: point WILOR_ROOT at a WiLoR checkout
UPSTREAM = Path(os.environ.get("WILOR_ROOT", ROOT / "third_party" / "WiLoR"))
MIRROR = np.diag([-1.0, 1.0, 1.0])


def nearest_so3(m):
    """Project a 3x3 matrix to the nearest proper rotation (det=+1)."""
    u, _, vt = np.linalg.svd(np.asarray(m, dtype=np.float64))
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def resolve_wrist_rot(global_orient, is_right):
    """MANO root rotation -> proper SO(3) in the camera frame.

    WiLoR runs only a RIGHT-hand model; left hands are processed as horizontally
    flipped crops, so the network output lives in the x-mirrored world. The true
    left-hand rotation is M R M with M = diag(-1,1,1) (det stays +1). For a left
    hand the rotation maps the x-mirrored right-MANO canonical frame (the left
    MANO frame) to the camera frame.
    """
    r = np.asarray(global_orient, dtype=np.float64).reshape(3, 3)
    if not is_right:
        r = MIRROR @ r @ MIRROR
    return nearest_so3(r)


def hand_message(side, conf, joints_cam, global_orient, intrinsics):
    """Build the per-hand dict of the wire format from real-hand-space data."""
    joints = np.asarray(joints_cam, dtype=np.float64)
    z = np.maximum(joints[:, 2], 1e-6)
    uv = np.stack([joints[:, 0] / z * intrinsics["fx"] + intrinsics["cx"],
                   joints[:, 1] / z * intrinsics["fy"] + intrinsics["cy"]], axis=1)
    return {"conf": float(conf), "wrist_pos": joints[0].tolist(),
            "wrist_rot": nearest_so3(global_orient).tolist(),
            "keypoints": joints.tolist(), "keypoints_2d": uv.tolist()}


class WiLoREngine:
    def __init__(self, device="cuda", fast=True, reference_preprocessing=False):
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False "
                               "(driver/library mismatch? use --device cpu)")
        self.torch, self.device = torch, device
        # 24 intra-op threads make the tiny CPU tensor ops (fp16 cast/collate) ~6x slower
        # (bimodal 4/24 ms prep stage); 4 is stable and fastest.
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.fast = bool(fast and device == "cuda")  # FP16+compile only on CUDA
        required = [UPSTREAM / p for p in ("pretrained_models/detector.pt",
                    "pretrained_models/wilor_final.ckpt", "mano_data/MANO_RIGHT.pkl")]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise RuntimeError("WiLoR assets missing: " + ", ".join(missing))
        os.chdir(UPSTREAM)  # upstream config uses ./mano_data
        for p in (str(UPSTREAM), str(TELEOP)):
            if p not in sys.path:
                sys.path.insert(0, p)
        import inspect
        if not hasattr(inspect, "getargspec"):
            inspect.getargspec = inspect.getfullargspec
        for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                            "object": object, "str": str, "unicode": str}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        from ultralytics import YOLO
        from wilor.models import load_wilor
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        import wilor.datasets.vitdet_dataset as dataset_module
        from wilor.utils import recursive_to
        import preprocessing
        import wilor.models.wilor as wilor_module
        # The model ctor builds pyrender/EGL visualisation renderers we never use; EGL init
        # fails without a working GPU driver, so stub them out.
        wilor_module.SkeletonRenderer = lambda *a, **k: None
        wilor_module.MeshRenderer = lambda *a, **k: None
        dataset_module.gaussian = (preprocessing.reference_gaussian if reference_preprocessing
                                   else preprocessing.gaussian_rgb)
        self.dataset_class, self.recursive_to = self._make_roi_dataset(ViTDetDataset, dataset_module), recursive_to
        original_load = torch.load

        def checkpoint_load(*a, **kw):
            kw.setdefault("weights_only", False)
            return original_load(*a, **kw)

        torch.load = checkpoint_load
        try:
            self.model, self.cfg = load_wilor(str(required[1]),
                                              str(UPSTREAM / "pretrained_models/model_config.yaml"))
            self.detector = YOLO(str(required[0])).to(device)
        finally:
            torch.load = original_load
        self.model = self.model.to(device).eval()
        if self.fast:
            torch.set_float32_matmul_precision("high")
            self.model = self.model.half()
            self.model.backbone = torch.compile(self.model.backbone)
            self.model.backbone.skip_blocks = True
        if self.detector.names != {0: "left", 1: "right"}:
            raise RuntimeError("Unexpected detector class mapping: " + repr(self.detector.names))
        print(f"[engine] loaded device={device} fast={self.fast}", flush=True)

    @staticmethod
    def _make_roi_dataset(base, dm):
        """ViTDetDataset whose antialias blur + crop only touch the hand ROI.

        Upstream copies and Gaussian-blurs the WHOLE frame (float64) per hand whenever the
        hand box is large (~20+ ms at 720p). The patch only samples a square around the box,
        so blurring that square (+ kernel radius margin) is numerically identical.
        """
        import cv2
        import torch

        class RoiDataset(base):
            def __getitem__(self, idx):
                center = self.center[idx].copy()
                bbox_size = dm.expand_to_aspect_ratio(
                    self.scale[idx] * 200, target_aspect_ratio=self.cfg.MODEL.get("BBOX_SHAPE", None)).max()
                size = self.img_size
                flip = self.right[idx] == 0
                img = self.img_cv2
                H, W = img.shape[:2]
                factor = bbox_size / size / 2.0
                sigma = (factor - 1) / 2 if factor > 1.1 else 0.0
                radius = int(4.0 * sigma + .5) if sigma else 0
                pad = int(bbox_size * 0.5 * 1.5 + radius + 3)   # 1.5: slack for the affine
                x0, x1 = max(0, int(center[0] - pad)), min(W, int(center[0] + pad) + 1)
                y0, y1 = max(0, int(center[1] - pad)), min(H, int(center[1] + pad) + 1)
                roi = img[y0:y1, x0:x1]
                if sigma:
                    roi = dm.gaussian(roi, sigma=sigma, channel_axis=2, preserve_range=True)
                patch, _ = dm.generate_image_patch_cv2(roi, center[0] - x0, center[1] - y0, bbox_size,
                                                       bbox_size, size, size, flip, 1.0, 0,
                                                       border_mode=cv2.BORDER_CONSTANT)
                patch = dm.convert_cvimg_to_tensor(patch[:, :, ::-1])
                for c in range(3):
                    patch[c] = (patch[c] - self.mean[c]) / self.std[c]
                if self.fp16:
                    patch = torch.from_numpy(patch).half()
                return {"img": patch, "personid": int(self.personid[idx]),
                        "box_center": self.center[idx].copy(), "box_size": bbox_size,
                        "img_size": 1.0 * np.array([W, H]), "right": self.right[idx].copy()}

        return RoiDataset

    def _sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()

    def warmup(self):
        t0 = time.monotonic()
        dtype = self.torch.float16 if self.fast else self.torch.float32
        size = self.cfg.MODEL.IMAGE_SIZE
        counts = (1, 2) if self.device == "cuda" else (1,)
        for n in counts:
            x = self.torch.zeros((n, 3, size, size), device=self.device, dtype=dtype)
            with self.torch.inference_mode():
                self.model({"img": x})
        self._sync()
        print(f"[engine] warmup {time.monotonic() - t0:.1f}s", flush=True)

    def infer(self, frame_bgr, intrinsics):
        """-> (hands dict {'left'|'right': hand dict}, timings dict)."""
        from camera_intrinsics import calibrated_crop_translation
        torch = self.torch
        self._sync()
        t0 = time.monotonic()
        boxes_all = self.detector(frame_bgr, conf=.3, verbose=False)[0].boxes
        boxes = boxes_all.xyxy.cpu().numpy()
        right = boxes_all.cls.cpu().numpy()
        conf = boxes_all.conf.cpu().numpy()
        # Keep the most confident detection per side (never drop both on a duplicate).
        keep = []
        for r in (0, 1):
            idx = [i for i in range(len(right)) if int(right[i]) == r]
            if idx:
                keep.append(max(idx, key=lambda i: conf[i]))
        boxes, right, conf = boxes[keep], right[keep], conf[keep]
        t1 = time.monotonic()
        hands = {}
        tp = tm = t1
        if len(keep):
            ds = self.dataset_class(self.cfg, frame_bgr, boxes, right, rescale_factor=2.,
                                    fp16=self.fast)
            batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
            tp = time.monotonic()
            batch = self.recursive_to(batch, self.device)
            with torch.inference_mode():
                out = self.model(batch)
            self._sync()
            tm = time.monotonic()
            cam = out["pred_cam"].float().clone()
            sgn = (2 * batch["right"] - 1)
            cam[:, 1] *= sgn
            trans = calibrated_crop_translation(
                cam.cpu().numpy(), batch["box_center"].float().cpu().numpy(),
                batch["box_size"].float().cpu().numpy(), intrinsics).astype(np.float64)
            kp = out["pred_keypoints_3d"].detach().float().cpu().numpy().astype(np.float64)
            go = out["pred_mano_params"]["global_orient"].detach().float().cpu().numpy()
            for i in range(len(keep)):
                is_right = bool(right[i] == 1)
                j = kp[i].copy()
                j[:, 0] *= 1.0 if is_right else -1.0   # undo the left-hand flip
                j += trans[i]
                R = resolve_wrist_rot(go[i].reshape(3, 3), is_right)
                hands["right" if is_right else "left"] = hand_message(
                    "right" if is_right else "left", conf[i], j, R, intrinsics)
        t2 = time.monotonic()
        return hands, {"detector_ms": (t1 - t0) * 1e3, "prep_ms": (tp - t1) * 1e3,
                       "wilor_ms": (tm - tp) * 1e3, "post_ms": (t2 - tm) * 1e3,
                       "total_ms": (t2 - t0) * 1e3}
