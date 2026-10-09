#!/usr/bin/env python
"""Send start/stop to a running launch_xr_teleop.py started with --ipc (same as keys r / q).

    data/xr-env/bin/python xr_camera_teleop/xr_ctl.py start|stop|status|reset

reset (works without --ipc): SIGUSR1 to the launcher (pid file data/xr_run/launch_xr_teleop.pid):
clutch uncalibrated + arms driven back to the home posture q=0 in joint space (SafeG1_29_ArmIK).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "xr_teleoperate"))
import zmq  # noqa: E402


def main():
    cmd = (sys.argv[1] if len(sys.argv) > 1 else "status").lower()
    if cmd == "reset":
        import os
        import signal
        pid_file = Path(__file__).resolve().parents[1] / "data" / "xr_run" / "launch_xr_teleop.pid"
        try:
            pid = int(pid_file.read_text().strip())
            # stale pid file guard: SIGUSR1 would terminate an unrelated process
            if b"launch_xr_teleop.py" not in Path(f"/proc/{pid}/cmdline").read_bytes():
                raise OSError(f"pid {pid} is not launch_xr_teleop.py (stale {pid_file})")
            os.kill(pid, signal.SIGUSR1)
            print(f"reset sent to launch_xr_teleop.py (pid {pid})")
        except (OSError, ValueError) as e:
            print(f"no running launch_xr_teleop.py ({e})")
        return
    ctx = zmq.Context.instance()
    if cmd == "status":
        s = ctx.socket(zmq.SUB)
        s.connect("ipc://@xr_teleoperate_hb.ipc")
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        if s.poll(2000):
            print(s.recv_json())
        else:
            print("no heartbeat (is launch_xr_teleop.py running with --ipc?)")
        return
    name = {"start": "CMD_START", "stop": "CMD_STOP", "record": "CMD_RECORD_TOGGLE"}[cmd]
    r = ctx.socket(zmq.REQ)
    r.setsockopt(zmq.RCVTIMEO, 3000)
    r.connect("ipc://@xr_teleoperate_data.ipc")
    r.send_json({"reqid": f"ctl-{time.time()}", "cmd": name})
    print(r.recv_json())


if __name__ == "__main__":
    main()
