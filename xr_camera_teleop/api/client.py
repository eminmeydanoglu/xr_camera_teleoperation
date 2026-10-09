"""HandStreamClient: non-blocking latest-message reader (stdlib + pyzmq + msgpack)."""
import time

import msgpack
import zmq


class HandStreamClient:
    def __init__(self, addr="tcp://127.0.0.1:5590"):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)   # must be set before connect
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.connect(addr)
        self._last = None

    def latest(self, max_age=0.3):
        """Newest message, or None if nothing received or older than max_age seconds
        (age = now - t_publish on the shared host monotonic clock; never blocks)."""
        while True:
            try:
                raw = self._sock.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            self._last = (time.monotonic(), msgpack.unpackb(raw, raw=False))
        if self._last is None:
            return None
        t_rx, msg = self._last
        now = time.monotonic()
        # time.monotonic() is system-wide on Linux, so t_publish is comparable on the
        # same host; fall back to receipt time if the stamp looks foreign.
        age = now - msg.get("t_publish", t_rx)
        if not (-0.05 <= age < 3600.0):
            age = now - t_rx
        if max_age is not None and age > max_age:
            return None
        return msg

    def close(self):
        self._sock.close(0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
