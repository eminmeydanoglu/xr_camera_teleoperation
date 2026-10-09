#!/usr/bin/env python
"""End-to-end latency probe: step input on the hand stream -> sim joint response.

  latency_probe.py publish --step-at 12 --out step.json   # replaces replay.py; static hands, then a step
  latency_probe.py analyze trace.npz step.json            # trace from g1_dds_mujoco.py --trace

All timestamps are time.monotonic() (CLOCK_MONOTONIC, shared by processes on one host).
Reported (relative to the t_publish of the first stepped message):
  first lowcmd change : t_publish -> stream -> WiLoRTeleWrapper(filters, clutch) -> IK -> lowcmd rx at sim
  first joint motion  : ... -> PD -> joint moved > 5% of its final delta
  50% / 90% of the commanded final delta (includes the wrapper's One-Euro/rate limiter and PD dynamics)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def publish(a):
    import msgpack
    import zmq
    from xr_camera_teleop.teleop.synthetic_stream import REST, hand_message_entry, make_message, rest_rotation
    s = zmq.Context.instance().socket(zmq.PUB)
    s.bind(a.addr)
    time.sleep(0.5)
    t0 = time.monotonic()
    fid, t_step, period = 0, None, 1.0 / a.hz
    nxt = t0
    while time.monotonic() - t0 < a.duration:
        t = time.monotonic()
        stepped = t - t0 >= a.step_at
        off = np.array([0.0, 0.0, 0.0]) + (np.array(a.delta) if stepped else 0)
        hands = {sd: hand_message_entry(sd, REST[sd] + off, rest_rotation(sd)) for sd in ("left", "right")}
        msg = make_message(fid, t, hands)
        msg["source"] = "replay"
        fid += 1
        if stepped and t_step is None:
            t_step = t
            Path(a.out).write_text(json.dumps({"t_step_publish": t_step}))
        s.send(msgpack.packb(msg, use_bin_type=True))
        nxt += period
        time.sleep(max(0, nxt - time.monotonic()))


def analyze(a):
    z = np.load(a.trace)
    t_step = json.loads(Path(a.step).read_text())["t_step_publish"]
    ct, cq, st, sq = z["cmd_t"], z["cmd_q"], z["st_t"], z["st_q"]
    arm = slice(15, 29)
    base_c = cq[(ct < t_step) & (ct > t_step - 1.0)][:, arm].mean(0)
    base_s = sq[(st < t_step) & (st > t_step - 1.0)][:, arm].mean(0)
    after_c, after_s = cq[ct > t_step][:, arm], sq[st > t_step][:, arm]
    fin_c = cq[(ct > t_step + 3.0)][:, arm][:100].mean(0)           # settled command
    fin_s = sq[(st > t_step + 3.0)][:, arm][:100].mean(0)
    j = int(np.argmax(np.abs(fin_c - base_c)))                      # most-moving arm joint
    d_c = fin_c[j] - base_c[j]
    print(f"joint index {15 + j}: command {base_c[j]:+.3f} -> {fin_c[j]:+.3f} rad, state {base_s[j]:+.3f} -> {fin_s[j]:+.3f}")
    ncmd = np.abs(after_c - base_c).max(1)
    first = np.argmax(ncmd > 2e-3)
    print(f"first lowcmd change (>2 mrad any arm joint): {(ct[ct > t_step][first] - t_step) * 1e3:.1f} ms")
    ts_after = st[st > t_step]
    for name, frac in (("first joint motion (5% of cmd delta)", 0.05), ("50% of cmd delta", 0.5), ("90% of cmd delta", 0.9)):
        prog = (after_s[:, j] - base_s[j]) / d_c
        i = np.argmax(prog > frac)
        print(f"{name}: {(ts_after[i] - t_step) * 1e3:.1f} ms")
    progc = (after_c[:, j] - base_c[j]) / d_c
    tca = ct[ct > t_step]
    for frac in (0.5, 0.9):
        print(f"  (command itself reaches {int(frac*100)}% at {(tca[np.argmax(progc > frac)] - t_step) * 1e3:.1f} ms)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("publish")
    q.add_argument("--addr", default="tcp://127.0.0.1:5590")
    q.add_argument("--hz", type=float, default=30.0)
    q.add_argument("--step-at", type=float, default=12.0)
    q.add_argument("--duration", type=float, default=30.0)
    q.add_argument("--delta", type=float, nargs=3, default=[0.0, 0.0, 0.12], help="step in W [m]")
    q.add_argument("--out", default="step.json")
    r = sub.add_parser("analyze")
    r.add_argument("trace")
    r.add_argument("step")
    a = p.parse_args()
    publish(a) if a.cmd == "publish" else analyze(a)
