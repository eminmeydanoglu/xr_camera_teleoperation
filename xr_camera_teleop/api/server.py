#!/usr/bin/env python
"""WiLoR hand-tracking API server (run with data/wilor-venv).

capture thread (latest frame only) -> inference thread -> ZMQ PUB (msgpack)
                                          `-> overlay thread -> MJPEG
HTTP :8770  /  /overlay.mjpg  /latest.json  /health
"""
import argparse
import queue
import json
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import msgpack
import numpy as np
import zmq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from overlay import draw_overlay  # noqa: E402,F401  (shared with xr_camera_teleop/tools/render_session.py)

PAGE = b"""<!doctype html><meta charset=utf-8><title>WiLoR XR</title>
<body style="margin:0;background:#111;color:#ddd;font:14px system-ui">
<img src="/overlay.mjpg" style="max-width:100%;display:block">
<pre id=h style="margin:8px"></pre>
<script>setInterval(async()=>{try{h.textContent=JSON.stringify(await (await fetch('/health')).json(),null,1)}catch(e){}},1000)</script>"""


class Latest:
    """Single-slot mailbox: put overwrites, get blocks for a newer item."""

    def __init__(self):
        self.cv, self.item, self.seq = threading.Condition(), None, 0

    def put(self, item):
        with self.cv:
            self.item, self.seq = item, self.seq + 1
            self.cv.notify_all()

    def get(self, after, timeout=0.5):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != after, timeout)
            return (self.item, self.seq) if self.seq != after else (None, after)


class App:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.frames, self.overlay_in, self.jpeg = Latest(), Latest(), Latest()
        self.latest_msg = None
        self.stats = {"fps": 0.0, "infer_ms": 0.0, "latency_ms": 0.0, "frames": 0, "capture_fps": 0.0}
        self.error = None
        self.ready = False
        self.record = open(args.record, "w") if args.record else None
        self.video_in = queue.Queue(maxsize=90) if args.record_video else None  # no frame drops for raw video
        self.video_dropped = 0

    # -- threads ---------------------------------------------------------
    def capture_loop(self):
        n, t0 = 0, time.monotonic()
        try:
            while not self.stop.is_set():
                frame = self.source.read()
                if frame is None:
                    time.sleep(0.005)
                    continue
                n += 1
                now = time.monotonic()
                self.frames.put((n, now, frame))    # overwrite: latest only, no queue
                if self.video_in is not None:
                    try:
                        self.video_in.put_nowait((n, now, frame))
                    except queue.Full:
                        self.video_dropped += 1
                self.stats["capture_fps"] = n / max(now - t0, 1e-6) if n > 1 else 0.0
        except Exception as exc:
            self.error = f"capture: {exc!r}"
            self.stop.set()

    def inference_loop(self):
        seq, window = 0, []
        pub = zmq.Context.instance().socket(zmq.PUB)
        pub.setsockopt(zmq.SNDHWM, 2)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(self.args.zmq)
        k = {key: float(self.source.intrinsics[key]) for key in ("fx", "fy", "cx", "cy")}
        try:
            while not self.stop.is_set():
                item, seq = self.frames.get(seq)
                if item is None:
                    continue
                fid, t_cap, frame = item
                hands, tim = self.engine.infer(frame, self.source.intrinsics)
                msg = {"v": 1, "frame_id": fid, "t_capture": t_cap, "t_publish": 0.0,
                       "image_size": [frame.shape[1], frame.shape[0]], "intrinsics": k,
                       "source": "live", "hands": hands}
                if getattr(self.source, "max_hand_depth", None):
                    msg["max_hand_depth"] = self.source.max_hand_depth
                lv = self.source.level()
                if lv is not None:
                    msg["cam_R_level"] = lv
                msg["t_publish"] = time.monotonic()
                pub.send(msgpack.packb(msg, use_bin_type=True), zmq.NOBLOCK)
                window.append(msg["t_publish"])
                window = [t for t in window if msg["t_publish"] - t < 2.0]
                self.stats.update(fps=(len(window) - 1) / max(window[-1] - window[0], 1e-6)
                                  if len(window) > 1 else 0.0, infer_ms=tim["total_ms"],
                                  latency_ms=(msg["t_publish"] - t_cap) * 1e3, frames=fid)
                self.latest_msg = msg
                if self.record:
                    self.record.write(json.dumps(msg) + "\n")
                self.overlay_in.put((frame, msg, dict(self.stats)))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = f"inference: {exc!r}"
            self.stop.set()
        finally:
            pub.close()

    def overlay_loop(self):
        seq = 0
        while not self.stop.is_set():
            item, seq = self.overlay_in.get(seq)
            if item is None:
                continue
            frame, msg, stats = item
            ok, buf = cv2.imencode(".jpg", draw_overlay(frame, msg, stats),
                                   [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.jpeg.put(buf.tobytes())        # immutable bytes: each client gets its own copy ref

    def video_loop(self):
        """Raw camera recording (every captured frame, independent of inference)."""
        base = Path(self.args.record_video)
        base.parent.mkdir(parents=True, exist_ok=True)
        writer, stamps = None, open(base.with_suffix(".csv"), "w")
        stamps.write("video_index,frame_id,t_capture\n")
        idx = 0
        while not (self.stop.is_set() and self.video_in.empty()):
            try:
                fid, t_cap, frame = self.video_in.get(timeout=0.2)
            except queue.Empty:
                continue
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(base.with_suffix(".mkv")), cv2.VideoWriter_fourcc(*"MJPG"),
                                         30.0, (w, h))
                base.with_suffix(".json").write_text(json.dumps({
                    "intrinsics": self.source.intrinsics, "camera": self.source.info,
                    "fps_nominal": 30.0, "codec": "MJPG"}, default=str, indent=1))
                print(f"[server] recording camera to {base.with_suffix('.mkv')}", flush=True)
            writer.write(frame)
            stamps.write(f"{idx},{fid},{t_cap:.6f}\n")
            idx += 1
        if writer is not None:
            writer.release()
            lm = self.source.level_mean()
            if lm is not None:  # IMU gravity levelling, mean over the first second
                side = json.loads(base.with_suffix(".json").read_text())
                side["cam_R_level"] = lm
                base.with_suffix(".json").write_text(json.dumps(side, default=str, indent=1))
        stamps.close()
        print(f"[server] camera recording closed: {idx} frames, {self.video_dropped} dropped", flush=True)

    def health(self):
        m = self.latest_msg
        return {"ok": self.error is None and self.ready, "error": self.error, "device": self.args.device,
                "fast": self.engine.fast, "source": self.args.source, "camera": self.source.info,
                "intrinsics": self.source.intrinsics, **{k: round(v, 2) if isinstance(v, float) else v
                                                         for k, v in self.stats.items()},
                "last_frame_id": m["frame_id"] if m else None,
                "last_age_s": round(time.monotonic() - m["t_publish"], 3) if m else None,
                "hands": sorted(m["hands"]) if m else []}

    # -- lifecycle -------------------------------------------------------
    def start(self):
        from engine import WiLoREngine
        from sources import open_source
        a = self.args
        self.engine = WiLoREngine(device=a.device, fast=a.fast)
        self.engine.warmup()
        self.source = open_source(a.source)
        print("[server] source:", self.source.info, self.source.intrinsics, flush=True)
        self.threads = [threading.Thread(target=f, daemon=True)
                        for f in (self.capture_loop, self.inference_loop, self.overlay_loop)]
        if self.video_in is not None:
            self.threads.append(threading.Thread(target=self.video_loop))  # non-daemon: finish the file
        for t in self.threads:
            t.start()
        self.ready = True
        app = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/":
                    self._send(200, PAGE, "text/html")
                elif path == "/health":
                    self._send(200, json.dumps(app.health()).encode(), "application/json")
                elif path == "/latest.json":
                    self._send(200 if app.latest_msg else 204,
                               json.dumps(app.latest_msg).encode() if app.latest_msg else b"",
                               "application/json")
                elif path == "/overlay.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    seq = 0
                    try:
                        while not app.stop.is_set():
                            jpg, seq = app.jpeg.get(seq, 1.0)
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                else:
                    self._send(404, b'{"error":"not found"}', "application/json")

        self.httpd = ThreadingHTTPServer((a.host, a.http_port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        print(f"[server] ZMQ {a.zmq} | HTTP http://{a.host}:{a.http_port}/", flush=True)

    def run(self):
        self.start()
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        try:
            while not self.stop.is_set():
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        self.stop.set()
        self.httpd.shutdown()
        for t in self.threads:
            t.join(timeout=2 if t.daemon else 15)
        self.source.close()
        if self.record:
            self.record.close()
        if self.error:
            print("[server] ERROR", self.error, file=sys.stderr)
            return 1
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", default="zed", help="zed | omron | <camera index> | video file | image file")
    ap.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                    help="FP16 + compiled backbone (CUDA only)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--zmq", default="tcp://127.0.0.1:5590")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--http-port", type=int, default=8770)
    ap.add_argument("--record", default=None, help="write published messages to this .jsonl")
    ap.add_argument("--record-video", default=None,
                    help="raw camera recording base path: writes <base>.mkv (MJPG), <base>.csv (timestamps), "
                         "<base>.json (intrinsics; reused when replayed with --source <base>.mkv)")
    args = ap.parse_args()
    if args.record:
        args.record = str(Path(args.record).resolve())
    if args.record_video:
        args.record_video = str(Path(args.record_video).resolve())
    if args.source not in ("zed",) and Path(args.source).exists():
        args.source = str(Path(args.source).resolve())  # engine chdirs into WILOR_ROOT
    sys.exit(App(args).run())


if __name__ == "__main__":
    main()
