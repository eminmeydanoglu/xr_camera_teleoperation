"""WiLoR RGB overlay drawing (shared by api/server.py and tools/render_session.py)."""
import cv2
import numpy as np

CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10),
               (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18),
               (18, 19), (19, 20)]
COLORS = {"left": (80, 230, 100), "right": (230, 180, 60)}
AXIS_COLORS = ((0, 0, 255), (0, 255, 0), (255, 0, 0))  # x r, y g, z b (BGR)


def draw_overlay(frame, msg, stats):
    view = frame.copy()                     # never touch the capture buffer
    k = msg["intrinsics"]
    for side, h in msg["hands"].items():
        color = COLORS[side]
        uv = np.asarray(h["keypoints_2d"])
        z = np.asarray(h["keypoints"])[:, 2]
        ok = (z > 1e-3) & np.isfinite(uv).all(1)
        uvi = np.clip(np.nan_to_num(uv), -5000, 5000).astype(int)
        for a, b in CONNECTIONS:
            if ok[a] and ok[b]:
                cv2.line(view, tuple(uvi[a]), tuple(uvi[b]), color, 2, cv2.LINE_AA)
        for i in range(21):
            if ok[i]:
                cv2.circle(view, tuple(uvi[i]), 3, color, -1, cv2.LINE_AA)
        p0, R = np.asarray(h["wrist_pos"]), np.asarray(h["wrist_rot"])
        if p0[2] > 1e-3:
            for ax in range(3):
                q = p0 + R[:, ax] * 0.05
                if q[2] > 1e-3:
                    a = (int(p0[0] / p0[2] * k["fx"] + k["cx"]), int(p0[1] / p0[2] * k["fy"] + k["cy"]))
                    b = (int(q[0] / q[2] * k["fx"] + k["cx"]), int(q[1] / q[2] * k["fy"] + k["cy"]))
                    cv2.line(view, a, b, AXIS_COLORS[ax], 3, cv2.LINE_AA)
            cv2.putText(view, f"{side} {h['conf']:.2f} z={p0[2]:.2f}m", (uvi[0][0] + 6, uvi[0][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    cv2.putText(view, f"{stats['fps']:.1f} FPS | infer {stats['infer_ms']:.0f} ms | "
                f"latency {stats['latency_ms']:.0f} ms | {len(msg['hands'])} hand(s)", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv2.LINE_AA)
    return view
