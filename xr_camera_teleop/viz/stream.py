"""Non-blocking latest-message reader. Uses agent A's client when present."""
from __future__ import annotations

import time


class MiniClient:
    """Tiny SUB+CONFLATE reader against the contract (fallback)."""

    def __init__(self, addr="tcp://127.0.0.1:5590"):
        import zmq

        self._zmq = zmq
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.connect(addr)
        self._msg, self._t = None, 0.0

    def latest(self, max_age=0.3):
        import msgpack

        try:
            raw = self.sock.recv(self._zmq.NOBLOCK)
            self._msg, self._t = msgpack.unpackb(raw, raw=False), time.monotonic()
        except self._zmq.Again:
            pass
        if self._msg is None or time.monotonic() - self._t > max_age:
            return None
        return self._msg

    def close(self):
        self.sock.close(0)


def make_client(addr):
    try:
        from xr_camera_teleop.api.client import HandStreamClient

        return HandStreamClient(addr)
    except Exception:
        return MiniClient(addr)
