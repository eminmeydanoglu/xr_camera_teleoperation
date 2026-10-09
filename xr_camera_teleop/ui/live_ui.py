#!/usr/bin/env python
"""Live 3-panel web UI (run with data/xr-env):  LEFT WiLoR overlay | MIDDLE floating hands | RIGHT G1.

    data/xr-env/bin/python xr_camera_teleop/ui/live_ui.py            # then open http://127.0.0.1:8780/

Needs the hand server (wilor-venv):  data/wilor-venv/bin/python xr_camera_teleop/api/server.py --source zed
Pipeline (same as tools/render_session.py, live): ZMQ :5590 -> WiLoRTeleWrapper (ego clutch, filters)
-> G1_29_ArmIK -> Dex3 HandRetargeting -> G1Sim (kinematic MuJoCo, pelvis fixed); floating hands via
HandMapper (shared anchor). The left panel is the server's own /overlay.mjpg (embedded by the page).
Threads: pipeline (60 Hz, ZMQ conflated latest) | render (EGL offscreen, ~30 fps, MJPEG) | HTTP.
Camera tilt: the per-message cam_R_level (ZED IMU) is used automatically; the page can override it
with a manual pitch. Keys on the page: C calibrate, R reset, Space pause/engage.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

PW, PH = 640, 480
HIDE_AFTER, STALE_AFTER = 2.0, 0.3
SIDES = ("left", "right")

PAGE = """<!doctype html><meta charset=utf-8><title>WiLoR live</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#14161a;--fg:#e6e8eb;--mut:#9aa3ad;--ac:#4aa3ff;--ok:#5ad27a;--warn:#ffb454}
body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 12px;background:#1c2026}
button{background:#2a3038;color:var(--fg);border:1px solid #3a424c;border-radius:6px;padding:6px 12px;cursor:pointer}
button:hover{border-color:var(--ac)} button.on{background:#7a3b1e}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;padding:4px}
.p{background:#000;position:relative}.p h3{margin:0;padding:4px 8px;font-size:13px;background:#1c2026;color:var(--mut)}
.p img{width:100%;display:block;aspect-ratio:4/3;object-fit:contain;background:#000}
#st{padding:8px 12px;font:13px ui-monospace,monospace;line-height:1.6}
.ok{color:var(--ok)}.warn{color:var(--warn)} input[type=range]{vertical-align:middle}
@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
<header>
<button onclick="post('calibrate')">Calibrate now (C)</button>
<button onclick="post('reset')">Reset (R)</button>
<button id=eng onclick="post('engage')">Pause / engage (Space)</button>
<button id=rec onclick="post('record')">&#9679; Record 3-panel mp4 (V)</button>
<label>k <input id=k type=range min=0.3 max=1.5 step=0.05 value=0.8 oninput="kv.textContent=this.value" onchange="post('k',{k:+this.value})"> <span id=kv>0.8</span></label>
<label>Cam pitch <select id=pm onchange="pitch()"><option value=auto>auto (IMU)</option><option value=manual>manual</option></select>
<input id=pd type=number value=0 step=1 style="width:60px" onchange="pitch()"> deg (+ = looking down)</label>
</header>
<div class=grid>
<div class=p><h3>WiLoR (server overlay)</h3><img id=ov></div>
<div class=p><h3>Floating hands</h3><img src="/mid.mjpg"></div>
<div class=p><h3>G1 (xr_teleoperate IK + Dex3)</h3><img src="/g1.mjpg"></div>
</div>
<div id=st>...</div>
<script>
document.getElementById('ov').src='http://'+location.hostname+':__OVPORT__/overlay.mjpg';
async function post(a,b){await fetch('/api/'+a,{method:'POST',body:JSON.stringify(b||{})});}
function pitch(){post('pitch',{mode:pm.value,deg:+pd.value})}
document.addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
 if(e.key==='c'||e.key==='C')post('calibrate');else if(e.key==='r'||e.key==='R')post('reset');
 else if(e.key==='v'||e.key==='V')post('record');
 else if(e.key===' '){e.preventDefault();post('engage')}});
async function poll(){try{const s=await (await fetch('/api/status')).json();
 const h=x=>{const v=s.hands[x];return x+': '+(v.seen?'<span class=ok>seen</span>':'<span class=warn>'+(v.detected_ever?'stale':'never')+'</span>')};
 eng.className=s.engaged?'':'on';
 rec.className=s.recording?'on':''; rec.innerHTML=s.recording?'&#9632; Stop recording ('+s.recording.seconds.toFixed(0)+' s)':'&#9679; Record 3-panel mp4 (V)';
 st.innerHTML=(s.calibrated?'<span class=ok>CALIBRATED</span>':'<span class=warn>waiting: hold both hands still 1 s (both inside the frame, closer than 0.9 m)</span>')
 +(s.engaged?'':' | <span class=warn>PAUSED</span>')+' | '+h('left')+' '+h('right')
 +'<br>tilt: '+s.tilt.source+' pitch '+s.tilt.pitch.toFixed(1)+'&deg; roll '+s.tilt.roll.toFixed(1)+'&deg; yaw '+(s.tilt.yaw||0).toFixed(0)+'&deg; | k='+s.k
 +'<br>fps: WiLoR '+s.fps.overlay.toFixed(1)+' | hands '+s.fps.mid.toFixed(1)+' | G1 '+s.fps.g1.toFixed(1)+' | pipeline '+s.fps.pipeline.toFixed(0)+' Hz'
 +' | latency capture&rarr;render '+s.latency_ms.toFixed(0)+' ms (server '+s.server_latency_ms.toFixed(0)+' ms)'
 +(s.self_collision?'<br>self-collision: '+(s.self_collision.on?'<span class=ok>ON</span>'+(s.self_collision.mode&&s.self_collision.mode!='track'?' <span class=warn>'+(s.self_collision.mode=='home'?'returning home':'switching arm branch')+'</span>':'')+' constrained '+s.self_collision.constrained+' / '+s.self_collision.solves+' solves, gated '+s.self_collision.gated+', escapes '+s.self_collision.escapes+', homes '+s.self_collision.homes+', IK '+s.self_collision.last_ms.toFixed(1)+' ms':'<span class=warn>OFF</span>'):'')
 +(s.last_recording?'<br>last recording: '+s.last_recording:'')
 +(s.error?'<br><span class=warn>'+s.error+'</span>':'');}catch(e){st.textContent='UI backend unreachable'}}
setInterval(poll,500);poll();
</script>"""


def put_text(img, s, org, scale=0.55, color=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


class Latest:
    def __init__(self):
        self.cv, self.item, self.seq = threading.Condition(), None, 0

    def put(self, item):
        with self.cv:
            self.item, self.seq = item, self.seq + 1
            self.cv.notify_all()

    def get(self, after, timeout=1.0):
        with self.cv:
            self.cv.wait_for(lambda: self.seq != after, timeout)
            return (self.item, self.seq) if self.seq != after else (None, after)


class Rate:
    def __init__(self, win=2.0):
        self.t, self.win = [], win

    def tick(self):
        now = time.monotonic()
        self.t.append(now)
        while self.t and now - self.t[0] > self.win:
            self.t.pop(0)

    def hz(self):
        return (len(self.t) - 1) / (self.t[-1] - self.t[0]) if len(self.t) > 2 and self.t[-1] > self.t[0] else 0.0


class ClientProxy:
    """Wraps HandStreamClient; with a manual pitch override it hides the message's cam_R_level."""

    def __init__(self, client, ui):
        self.client, self.ui = client, ui

    def latest(self, max_age=0.3):
        m = self.client.latest(max_age=max_age)
        if m is not None and self.ui.pitch_mode == "manual" and "cam_R_level" in m:
            m = {k: v for k, v in m.items() if k != "cam_R_level"}
        return m

    def close(self):
        self.client.close()


class LiveUI:
    def __init__(self, args):
        self.a = args
        self.lock = threading.Lock()
        self.rec_proc = self.rec_path = self.rec_left = self.last_rec = None
        self.rec_t0 = 0.0
        self.rec_lock = threading.Lock()
        self.stop = threading.Event()
        self.pitch_mode, self.pitch_deg = "auto", 0.0
        self.k = args.k
        self.engaged = True
        self.cmds = []
        self.error = None
        self.fps = {"mid": Rate(), "g1": Rate(), "pipe": Rate()}
        self.mid_jpg, self.g1_jpg = Latest(), Latest()
        self.latency_ms = 0.0
        self.server_latency_ms = 0.0
        self.overlay_fps = 0.0
        self.tilt = {"source": "none", "pitch": 0.0, "roll": 0.0}
        # shared pipeline -> render state
        self.hand_res = {h: None for h in SIDES}
        self.hand_state = {h: "hidden" for h in SIDES}
        self.qpos = None
        self.t_cap = None
        self.calibrated = False
        self.swapped = False
        self.hand_seen = {h: {"seen": False, "detected_ever": False} for h in SIDES}
        self.ready = threading.Event()
        self.ik = None

    # ------------------------------------------------------------ pipeline thread
    def pipeline(self):
        try:
            self._pipeline()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error = f"pipeline crashed: {e!r}"

    def _pipeline(self):
        from xr_camera_teleop.api.client import HandStreamClient
        from xr_camera_teleop.teleop import xrt_compat
        from xr_camera_teleop.teleop.camlevel import level_rotation
        from xr_camera_teleop.teleop.clutch import neutral_from_ik_fk
        from xr_camera_teleop.teleop.g1_mujoco_teleop import G1Sim, solve_hand
        from xr_camera_teleop.teleop.wilor_tv_wrapper import WiLoRTeleWrapper
        from xr_camera_teleop.viz.mapping import WAIST_DROP, HandMapper, robot_anchors

        a = self.a
        ik = xrt_compat.make_arm_ik(self_collision=a.self_collision)
        self.ik = ik
        hr = xrt_compat.make_hand_retargeting()
        arm_names = list(ik.reduced_robot.model.names[1:])
        hand_names = {"left": list(hr.left_retargeting_joint_names), "right": list(hr.right_retargeting_joint_names)}
        self.sim = G1Sim(arm_names, hand_names)
        wrapper = WiLoRTeleWrapper(client=ClientProxy(HandStreamClient(a.addr), self), k=a.k,
                                   neutral=neutral_from_ik_fk(ik), auto_calibrate=True)
        xrt_compat.connect_ik(ik, wrapper)       # Reset / L-R swap -> arms back to q=0 in joint space
        waist_h = 1.5 - WAIST_DROP
        self.waist_h = waist_h
        mapper = HandMapper("relative", a.k, 1.5, neutral_rot="ref", anchors=robot_anchors(waist_h), max_depth=0.9)
        last_cam, last_seen = {}, {h: -1e9 for h in SIDES}
        cur_dq = np.zeros(14)
        last_fid = None
        hand_q = {}
        period = 1 / 60.0
        nxt = prev = time.monotonic()
        self.ready.set()
        while not self.stop.is_set():
            now = time.monotonic()
            dt = float(np.clip(now - prev, 1 / 240, 0.1))
            prev = now
            with self.lock:
                cmds, self.cmds = self.cmds, []
            for c, v in cmds:
                if c == "calibrate":
                    wrapper.calibrate()
                elif c == "reset":
                    wrapper.reset()
                    mapper.clear()
                elif c == "engage":
                    self.engaged = not self.engaged
                    wrapper.set_engaged(self.engaged)
                elif c == "k":
                    self.k = float(np.clip(v, 0.3, 1.5))
                    wrapper.set_k(self.k)
                    mapper.scale = self.k
            wrapper.cam_pitch_deg = self.pitch_deg
            td = wrapper.get_tele_data()
            msg = wrapper.last_msg
            if msg is not None and msg.get("frame_id") != last_fid:      # new frame: floating hands + fingers
                last_fid = msg.get("frame_id")
                R_lv = level_rotation(msg, self.pitch_deg)
                mapper.R_level = wrapper.effective_level()      # incl. operator heading (auto yaw)
                d = R_lv[1]
                src = "IMU" if msg.get("cam_R_level") is not None else ("manual" if self.pitch_mode == "manual"
                                                                      else "fallback (no IMU)")
                self.tilt = {"source": src, "pitch": float(np.degrees(np.arctan2(d[2], d[1]))),
                             "roll": float(np.degrees(np.arctan2(d[0], d[1]))),
                             "yaw": float(np.degrees(wrapper.yaw))}
                self.t_cap = float(msg.get("t_capture", now))
                self.server_latency_ms = (float(msg.get("t_publish", now)) - self.t_cap) * 1e3
                for h, dd in msg.get("hands", {}).items():
                    if dd["wrist_pos"][2] > 0.9:           # same gate as the wrapper: not the operator's hand
                        continue
                    last_cam[h] = (np.asarray(dd["wrist_pos"], float), np.asarray(dd["wrist_rot"], float),
                                   np.asarray(dd["keypoints"], float))
                    last_seen[h] = now
                    self.hand_seen[h]["detected_ever"] = True
                if td.motion_data_ready:
                    for s, hp in (("left", td.left_hand_pos), ("right", td.right_hand_pos)):
                        q = solve_hand(hr, s, hp)
                        if np.all(np.isfinite(q)):
                            hand_q[s] = q
            cal = wrapper.clutch["left"].calibrated and wrapper.clutch["right"].calibrated
            if cal and not mapper.calibrated and all(h in last_cam for h in SIDES):
                mapper.calibrate({h: (last_cam[h][0], last_cam[h][1]) for h in SIDES})
            if not cal and mapper.calibrated:
                mapper.clear()
            res, st = {}, {}
            for h in SIDES:
                age = now - last_seen[h]
                r = mapper.map(h, *last_cam[h]) if (h in last_cam and age <= HIDE_AFTER and mapper.calibrated) else None
                res[h] = r
                st[h] = "hidden" if r is None else ("live" if age < STALE_AFTER else "stale")
                self.hand_seen[h]["seen"] = age < wrapper.hand_timeout
            # G1
            sol_q, _ = ik.solve_ik(td.left_wrist_pose, td.right_wrist_pose, self.sim.cur_arm.copy(), cur_dq)
            arm_q = np.asarray(sol_q, float) if np.all(np.isfinite(sol_q)) else None
            with self.lock:
                prev_arm = self.sim.cur_arm.copy()
                self.sim.step_targets(arm_q, hand_q, dt)
                cur_dq = (self.sim.cur_arm - prev_arm) / dt
                self.qpos = self.sim.data.qpos.copy()
                self.hand_res, self.hand_state, self.calibrated = res, st, cal
                self.swapped = wrapper.swapped
            self.fps["pipe"].tick()
            nxt += period
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            else:
                nxt = time.monotonic()
        wrapper.close()

    # ------------------------------------------------------------ render thread
    def render(self):
        try:
            self._render()
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error = f"render crashed: {e!r}"

    def _render(self):
        import mujoco
        from xr_camera_teleop.viz.hands_scene import HandsScene, default_camera

        self.ready.wait()
        while not hasattr(self, "waist_h"):
            time.sleep(0.05)
        sim = self.sim
        model = sim.model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, PW)
        model.vis.global_.offheight = max(model.vis.global_.offheight, PH)
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        scene = HandsScene(1.5, waist_height=self.waist_h)
        scene.model.vis.global_.offwidth = max(scene.model.vis.global_.offwidth, PW)
        scene.model.vis.global_.offheight = max(scene.model.vis.global_.offheight, PH)
        r_g1 = mujoco.Renderer(model, PH, PW)
        r_h = mujoco.Renderer(scene.model, PH, PW)

        def cam(lookat, dist, az, el):
            c = mujoco.MjvCamera()
            c.lookat[:] = lookat
            c.distance, c.azimuth, c.elevation = dist, az, el
            return c
        cam_g1 = cam([0.15, 0.0, 1.0], 1.15, 150, -8)
        look, dist, az, el = default_camera(1.5)
        cam_h = cam(look - np.array([0.0, 0.0, 0.15]), dist * 1.5, az, el)
        period = 1 / self.a.fps
        nxt = time.monotonic()
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        while not self.stop.is_set():
            with self.lock:
                qpos, res, st, cal = self.qpos, dict(self.hand_res), dict(self.hand_state), self.calibrated
                t_cap = self.t_cap
            if qpos is None:
                time.sleep(0.05)
                continue
            for h in ("left", "right"):
                scene.set_state(h, st[h])
                if res[h] is not None:
                    scene.pose(h, *res[h])
            scene.forward()
            r_h.update_scene(scene.data, cam_h)
            mid = cv2.cvtColor(r_h.render(), cv2.COLOR_RGB2BGR)
            put_text(mid, "calibrated" if cal else "waiting: hold both hands still 1 s", (10, PH - 12), 0.55,
                     (120, 255, 120) if cal else (120, 200, 255))
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            r_g1.update_scene(data, cam_g1)
            g1 = cv2.cvtColor(r_g1.render(), cv2.COLOR_RGB2BGR)
            if not self.engaged:
                put_text(g1, "PAUSED", (10, 28), 0.8, (80, 160, 255))
            self.mid_jpg.put(cv2.imencode(".jpg", mid, enc)[1].tobytes())
            self.g1_jpg.put(cv2.imencode(".jpg", g1, enc)[1].tobytes())
            self.fps["mid"].tick()
            self.fps["g1"].tick()
            if t_cap is not None:
                self.latency_ms = (time.monotonic() - t_cap) * 1e3
            nxt += period
            slp = nxt - time.monotonic()
            if slp > 0:
                time.sleep(slp)
            else:
                nxt = time.monotonic()

    # ------------------------------------------------------------ status / http
    def health_loop(self):
        while not self.stop.is_set():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.a.server_port}/health", timeout=1) as r:
                    self.overlay_fps = float(json.loads(r.read()).get("fps", 0.0))
            except Exception:  # noqa: BLE001
                self.overlay_fps = 0.0
            time.sleep(1.0)

    def status(self):
        return {"calibrated": self.calibrated, "swapped": self.swapped, "engaged": self.engaged, "k": self.k,
                "hands": {h: dict(self.hand_seen[h]) for h in SIDES}, "tilt": self.tilt,
                "pitch_mode": self.pitch_mode, "latency_ms": self.latency_ms,
                "server_latency_ms": self.server_latency_ms, "error": self.error,
                "recording": ({"path": self.rec_path, "seconds": time.monotonic() - self.rec_t0}
                              if self.rec_proc is not None else None),
                "last_recording": self.last_rec,
                "self_collision": (None if self.ik is None else
                                   dict(getattr(self.ik, "stats", {}), on=hasattr(self.ik, "stats"),
                                        mode=getattr(self.ik, "mode", None))),
                "fps": {"overlay": self.overlay_fps, "mid": self.fps["mid"].hz(), "g1": self.fps["g1"].hz(),
                        "pipeline": self.fps["pipe"].hz()}}

    # ------------------------------------------------------------ 3-panel recording
    def toggle_record(self):
        with self.rec_lock:
            if self.rec_proc is None:
                out = ROOT / "data/sessions" / time.strftime("live_%Y%m%d_%H%M%S.mp4")
                out.parent.mkdir(parents=True, exist_ok=True)
                self.rec_path, self.rec_t0, self.rec_left = str(out), time.monotonic(), None
                self.rec_proc = subprocess.Popen(
                    ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                     "-s", f"{3 * PW}x{PH + 40}", "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                     "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                    stdin=subprocess.PIPE)
                threading.Thread(target=self._rec_overlay_reader, args=(self.rec_proc,), daemon=True).start()
                threading.Thread(target=self._rec_writer, args=(self.rec_proc,), daemon=True).start()
                print("[live_ui] recording ->", out, flush=True)
            else:
                self._rec_close()

    def _rec_close(self):
        proc, self.rec_proc = self.rec_proc, None
        if proc is None:
            return
        time.sleep(0.1)                       # let the writer finish its current frame
        try:
            proc.stdin.close()
            proc.wait(timeout=20)
        except Exception:  # noqa: BLE001
            proc.kill()
        self.last_rec = self.rec_path
        print("[live_ui] recording saved:", self.rec_path, flush=True)

    def _rec_overlay_reader(self, proc):
        """Persistent read of the server's overlay MJPEG while recording (left panel)."""
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{self.a.server_port}/overlay.mjpg", timeout=3)
            buf = b""
            while self.rec_proc is proc and not self.stop.is_set():
                buf += r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                j = buf.rfind(b"\xff\xd9")
                i = buf.rfind(b"\xff\xd8", 0, j) if j >= 0 else -1
                if i >= 0:
                    img = cv2.imdecode(np.frombuffer(buf[i:j + 2], np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self.rec_left = cv2.resize(img, (PW, PH))
                    buf = buf[j + 2:]
        except Exception as exc:  # noqa: BLE001
            print("[live_ui] overlay reader:", exc, flush=True)

    def _rec_writer(self, proc):
        t_next = time.monotonic()
        while self.rec_proc is proc and not self.stop.is_set():
            left = self.rec_left if self.rec_left is not None else np.zeros((PH, PW, 3), np.uint8)
            try:
                proc.stdin.write(self.compose(left).tobytes())
            except (BrokenPipeError, ValueError, OSError):
                break
            t_next += 1 / 30
            time.sleep(max(0.0, t_next - time.monotonic()))
            if time.monotonic() - t_next > 0.5:   # fell behind: resync instead of bursting
                t_next = time.monotonic()

    def snapshot(self):
        """Composite JPEG (overlay frame from the server + both panels + status) for headless checks."""
        try:
            raw = urllib.request.urlopen(f"http://127.0.0.1:{self.a.server_port}/overlay.mjpg", timeout=2).read(400000)
            i = raw.find(b"\xff\xd8")
            j = raw.find(b"\xff\xd9", i)
            left = cv2.imdecode(np.frombuffer(raw[i:j + 2], np.uint8), cv2.IMREAD_COLOR)
            left = cv2.resize(left, (PW, PH))
        except Exception:  # noqa: BLE001
            left = np.zeros((PH, PW, 3), np.uint8)
        return cv2.imencode(".jpg", self.compose(left))[1].tobytes()

    def compose(self, left):
        """left | floating hands | G1 + status bar, BGR (PH+40, 3*PW)."""
        dec = lambda L: cv2.imdecode(np.frombuffer(L.item or b"", np.uint8), cv2.IMREAD_COLOR) \
            if L.item else np.zeros((PH, PW, 3), np.uint8)  # noqa: E731
        s = self.status()
        bar = np.full((40, 3 * PW, 3), 30, np.uint8)
        put_text(bar, f"{'CAL' if s['calibrated'] else 'wait'} | tilt {s['tilt']['source']} p{s['tilt']['pitch']:.1f} "
                      f"r{s['tilt']['roll']:.1f} yaw{s['tilt'].get('yaw', 0.0):.0f} | fps ov {s['fps']['overlay']:.0f} hands {s['fps']['mid']:.0f} "
                      f"g1 {s['fps']['g1']:.0f} | lat {s['latency_ms']:.0f}ms", (8, 26), 0.6)
        if s["swapped"]:
            put_text(bar, "L/R SWAP -> NEUTRAL", (3 * PW - 330, 26), 0.7, (60, 60, 255))
        if self.rec_proc is not None:
            put_text(bar, "REC", (3 * PW - 60, 26), 0.7, (60, 60, 255))
        return np.concatenate([np.concatenate([left, dec(self.mid_jpg), dec(self.g1_jpg)], 1), bar], 0)

    def serve(self):
        ui = self
        page = PAGE.replace("__OVPORT__", str(self.a.server_port)).encode()

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

            def _stream(self, L):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                seq = 0
                try:
                    while not ui.stop.is_set():
                        jpg, seq = L.get(seq, 1.0)
                        if jpg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                p = self.path.split("?")[0]
                if p == "/":
                    self._send(200, page, "text/html; charset=utf-8")
                elif p == "/mid.mjpg":
                    self._stream(ui.mid_jpg)
                elif p == "/g1.mjpg":
                    self._stream(ui.g1_jpg)
                elif p == "/api/status":
                    self._send(200, json.dumps(ui.status()).encode(), "application/json")
                elif p == "/snapshot.jpg":
                    self._send(200, ui.snapshot(), "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(n) or b"{}")
                except ValueError:
                    body = {}
                p = self.path.split("?")[0]
                if p == "/api/record":
                    threading.Thread(target=ui.toggle_record, daemon=True).start()
                elif p in ("/api/calibrate", "/api/reset", "/api/engage"):
                    with ui.lock:
                        ui.cmds.append((p[5:], None))
                elif p == "/api/k":
                    with ui.lock:
                        ui.cmds.append(("k", body.get("k", ui.k)))
                elif p == "/api/pitch":
                    ui.pitch_mode = "manual" if body.get("mode") == "manual" else "auto"
                    ui.pitch_deg = float(body.get("deg", 0.0))
                else:
                    return self._send(404, b"not found", "text/plain")
                self._send(200, b"{}", "application/json")

        self.httpd = ThreadingHTTPServer((self.a.host, self.a.port), H)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def run(self):
        for f in (self.pipeline, self.render, self.health_loop):
            threading.Thread(target=f, daemon=True).start()
        self.serve()
        print(f"[live_ui] http://{self.a.host}:{self.a.port}/   (hand server expected on :{self.a.server_port}, "
              f"ZMQ {self.a.addr})", flush=True)
        try:
            while not self.stop.is_set():
                time.sleep(0.3)
        except KeyboardInterrupt:
            pass
        self._rec_close()                 # finish the mp4 before the hard exit
        self.stop.set()
        time.sleep(0.2)
        os._exit(0)     # avoid noisy EGL teardown


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--addr", default="tcp://127.0.0.1:5590")
    ap.add_argument("--server-port", type=int, default=8770, help="hand server HTTP port (overlay.mjpg)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to open the UI from another machine")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--k", type=float, default=0.8)
    ap.add_argument("--fps", type=float, default=30.0, help="target render fps of the two MuJoCo panels")
    ap.add_argument("--dds", action="store_true", help="(not implemented) show the DDS sim instead of the kinematic G1")
    ap.add_argument("--no-self-collision", dest="self_collision", action="store_false", default=None,
                    help="original G1_29_ArmIK without self-collision protection (default: env "
                         "WILOR_XR_SELF_COLLISION, on)")
    a = ap.parse_args(argv)
    if a.dds:
        ap.error("--dds is not implemented; the kinematic G1Sim is used")
    LiveUI(a).run()


if __name__ == "__main__":
    main()
