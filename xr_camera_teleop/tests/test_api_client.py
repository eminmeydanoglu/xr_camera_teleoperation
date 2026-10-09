import sys
import threading
import time
from pathlib import Path

import msgpack
import pytest
import zmq

API = Path(__file__).resolve().parents[1] / "api"
sys.path.insert(0, str(API))
import replay  # noqa: E402
from client import HandStreamClient  # noqa: E402
from test_api_schema import validate  # noqa: E402


def free_addr():
    s = zmq.Context.instance().socket(zmq.REP)
    port = s.bind_to_random_port("tcp://127.0.0.1")
    s.close(0)
    return f"tcp://127.0.0.1:{port}"


class Args:
    synthetic, file, hz, duration, loop, realtime = True, None, 60.0, 1.5, False, False


def start_replay(**kw):
    a = Args()
    a.addr = free_addr()
    for k, v in kw.items():
        setattr(a, k, v)
    th = threading.Thread(target=replay.run, args=(a,), daemon=True)
    th.start()
    return a, th


def wait_msg(c, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        m = c.latest()
        if m:
            return m
        time.sleep(0.01)
    return None


def test_client_none_before_data_and_nonblocking():
    c = HandStreamClient(free_addr())
    t = time.monotonic()
    assert c.latest() is None
    assert time.monotonic() - t < 0.05
    c.close()


def test_client_receives_conformant_latest_and_is_conflated():
    a, th = start_replay()
    c = HandStreamClient(a.addr)
    m = wait_msg(c)
    assert m is not None
    validate(m)
    time.sleep(0.3)   # many messages pile up; CONFLATE must return only a recent one
    m2 = c.latest()
    assert m2["frame_id"] > m["frame_id"] + 3
    assert m2["frame_id"] >= m["frame_id"]
    c.close()
    th.join(3)


def test_client_max_age():
    a, th = start_replay(duration=0.5)
    c = HandStreamClient(a.addr)
    assert wait_msg(c) is not None
    th.join(3)               # publisher gone; stream stale
    time.sleep(0.45)
    assert c.latest(max_age=0.2) is None
    assert c.latest(max_age=10.0) is not None
    c.close()


def test_file_replay(tmp_path):
    import json
    path = tmp_path / "rec.jsonl"
    with open(path, "w") as f:
        for i in range(5):
            f.write(json.dumps({**replay.synthetic_message(i, i * 0.1), "source": "live"}) + "\n")
    a, th = start_replay(synthetic=False, file=str(path), duration=None, hz=100.0)
    c = HandStreamClient(a.addr)
    got = wait_msg(c)
    if got is not None:
        validate(got)
        assert got["source"] == "replay"
    c.close()
    th.join(3)


def test_overlay_does_not_mutate_input():
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    import server
    frame = np.zeros((720, 1280, 3), np.uint8)
    before = frame.copy()
    msg = replay.synthetic_message(1, 1.0)
    out = server.draw_overlay(frame, msg, {"fps": 30, "infer_ms": 10, "latency_ms": 20})
    assert (frame == before).all() and out is not frame and out.any()
