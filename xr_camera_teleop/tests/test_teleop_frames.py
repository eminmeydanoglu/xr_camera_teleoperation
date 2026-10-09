"""Frame conventions: WiLoR camera -> W -> xr_teleoperate TeleData."""
import numpy as np
import pytest

from xr_camera_teleop.teleop import synthetic_stream as syn
from xr_camera_teleop.teleop.clutch import HandClutch
from xr_camera_teleop.teleop.mano_to_webxr import mano21_to_webxr25
from xr_camera_teleop.teleop.wilor_tv_wrapper import WiLoRTeleWrapper, fallback_neutral

tvw = pytest.importorskip("xr_camera_teleop.teleop.xrt_compat").load_tv_wrapper()


class FakeClient:
    def __init__(self):
        self.msg = None

    def latest(self, max_age=0.3):
        return self.msg


class Clock:
    t = 100.0

    def __call__(self):
        return self.t


def make(**kw):
    cl, ck = FakeClient(), Clock()
    w = WiLoRTeleWrapper(client=cl, clock=ck, use_filters=False, k=kw.pop("k", 1.0),
                         neutral=fallback_neutral(), **kw)
    return w, cl, ck


_fid = [0]


def push(cl, ck, poses, grip=0.0, pinch=0.0, dt=0.033):
    ck.t += dt
    _fid[0] += 1
    hands = {s: syn.hand_message_entry(s, p, R, grip, pinch) for s, (p, R) in poses.items()}
    cl.msg = syn.make_message(_fid[0], ck.t, hands)


def rest():
    return {s: (syn.REST[s].copy(), syn.rest_rotation(s)) for s in ("left", "right")}


def test_cam_to_world_axes():
    # forward in camera (+z_cam) -> +x_W; camera right (+x) -> -y_W; camera down (+y) -> -z_W
    assert np.allclose(syn.R_W_CAM @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(syn.R_W_CAM @ [1, 0, 0], [0, -1, 0])
    assert np.allclose(syn.R_W_CAM @ [0, 1, 0], [0, 0, -1])


@pytest.mark.parametrize("axis_cam,expect_W", [([0, 0, 0.1], [0.1, 0, 0]),      # forward
                                              ([-0.1, 0, 0], [0, 0.1, 0]),     # left
                                              ([0, -0.1, 0], [0, 0, 0.1])])    # up
def test_operator_motion_maps_to_robot_waist_frame(axis_cam, expect_W):
    w, cl, ck = make(k=1.0)
    poses = rest()
    push(cl, ck, poses)
    w.calibrate()
    w.get_tele_data()
    td0 = w.get_tele_data()
    d_W = syn.R_W_CAM @ np.array(axis_cam)          # operator delta expressed in W
    poses2 = {s: (p + d_W, R) for s, (p, R) in poses.items()}
    push(cl, ck, poses2)
    td1 = w.get_tele_data()
    for T0, T1 in ((td0.left_wrist_pose, td1.left_wrist_pose), (td0.right_wrist_pose, td1.right_wrist_pose)):
        np.testing.assert_allclose(T1[:3, 3] - T0[:3, 3], expect_W, atol=1e-9)
        np.testing.assert_allclose(T1[:3, :3], T0[:3, :3], atol=1e-6)


def test_no_jump_at_calibration_and_neutral_output():
    w, cl, ck = make(mapping="per_hand")
    push(cl, ck, rest())
    td = w.get_tele_data()                        # uncalibrated: neutral
    n = fallback_neutral()
    np.testing.assert_allclose(td.left_wrist_pose, n[0])
    w.calibrate()
    td = w.get_tele_data()
    np.testing.assert_allclose(td.left_wrist_pose, n[0], atol=1e-9)
    np.testing.assert_allclose(td.right_wrist_pose, n[1], atol=1e-9)
    assert td.motion_data_ready


def test_rotation_about_world_z_is_transferred():
    w, cl, ck = make()
    poses = rest()
    push(cl, ck, poses)
    w.calibrate()
    td0 = w.get_tele_data()
    a = 0.4
    Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    push(cl, ck, {s: (p, Rz @ R) for s, (p, R) in poses.items()})
    td1 = w.get_tele_data()
    np.testing.assert_allclose(td1.left_wrist_pose[:3, :3], Rz @ td0.left_wrist_pose[:3, :3], atol=1e-6)


def test_hand_pos_matches_tv_wrapper_math():
    """Reproduce tv_wrapper.get_tele_data hand_pos pipeline (lines 351-376) from a simulated
    OpenXR-basis world hand + arm pose, compare with our hand_pos."""
    for side in ("left", "right"):
        w, cl, ck = make()
        # arbitrary world pose of the hand in W (robot basis)
        rng = np.random.default_rng(3)
        ang = rng.normal(size=3) * 0.5
        from xr_camera_teleop.teleop.filters import rot_exp
        R_W = rot_exp(ang) @ syn.rest_rotation(side)
        p_W = np.array([0.5, 0.1 if side == "left" else -0.1, -0.2])
        grip = 0.6
        poses = rest()
        poses[side] = (p_W, R_W)
        push(cl, ck, poses, grip=grip)
        td = w.get_tele_data()
        ours = td.left_hand_pos if side == "left" else td.right_hand_pos

        # --- reference: what TeleVuer would deliver for the same physical hand
        kp_local = syn.canonical_hand_mano21(side, grip)
        # we need the *keypoint-derived* frame, same as the wrapper (canonical hand has ~6deg offset)
        from xr_camera_teleop.teleop.mano_to_webxr import keypoint_wrist_frame
        Rk, _ = keypoint_wrist_frame(kp_local, side)
        R_arm_robot = R_W @ Rk                                    # IPxr arm frame, robot basis
        kpW = kp_local @ R_W.T + p_W
        w25 = mano21_to_webxr25(kpW)
        T_RO, T_OR = tvw.T_ROBOT_OPENXR, tvw.T_OPENXR_ROBOT
        arm_robot = np.eye(4)
        arm_robot[:3, :3], arm_robot[:3, 3] = R_arm_robot, w25[0]
        arm_xr = T_OR @ arm_robot @ T_RO                          # inverse of tv_wrapper.py:325
        pos_xr = (T_OR @ np.vstack([w25.T, np.ones(25)]))[:3].T   # OpenXR-basis world positions
        # ---- verbatim pipeline of tv_wrapper.py:326,351-376
        arm_robot2 = T_RO @ arm_xr @ T_OR
        h = np.concatenate([pos_xr.T, np.ones((1, 25))])
        h = T_RO @ h
        h = tvw.fast_mat_inv(arm_robot2) @ h
        ref = (tvw.T_TO_UNITREE_HAND @ h)[0:3, :].T
        np.testing.assert_allclose(ours, ref, atol=1e-9)
        assert np.allclose(ours[0], 0)


def test_hand_pos_invariant_to_wrist_pose_and_closes():
    w, cl, ck = make()
    poses = rest()
    push(cl, ck, poses, grip=0.0)
    open_pos = w.get_tele_data().right_hand_pos
    push(cl, ck, {s: (p + [0.1, 0.1, 0.1], R) for s, (p, R) in poses.items()}, grip=0.0)
    np.testing.assert_allclose(w.get_tele_data().right_hand_pos, open_pos, atol=1e-9)
    push(cl, ck, poses, grip=1.0)
    td = w.get_tele_data()
    assert td.right_hand_squeezeValue > 0.8 and td.right_hand_squeeze
    # fingertips move toward the wrist/palm in the Unitree hand frame
    assert np.linalg.norm(td.right_hand_pos[9]) < np.linalg.norm(open_pos[9])
    push(cl, ck, poses, pinch=1.0)
    td = w.get_tele_data()
    assert td.right_hand_pinch and td.right_hand_pinchValue < 2.5


def test_hand_loss_holds_last_valid_and_timeouts():
    w, cl, ck = make(k=1.0, stale_timeout=1.0)
    poses = rest()
    push(cl, ck, poses)
    w.calibrate()
    w.get_tele_data()
    push(cl, ck, {s: (p + [0.05, 0, 0], R) for s, (p, R) in poses.items()}, grip=0.5)
    a = w.get_tele_data()
    cl.msg = syn.make_message(999, ck.t + 0.1, {})        # hands lost
    ck.t += 0.5
    b = w.get_tele_data()
    np.testing.assert_allclose(b.left_wrist_pose, a.left_wrist_pose)
    np.testing.assert_allclose(b.left_hand_pos, a.left_hand_pos)
    assert not np.isnan(b.left_hand_pos).any()
    ck.t += 1.5
    c = w.get_tele_data()
    assert w.status()["left"]["stale"]
    # stale: drifts back towards the robot neutral instead of freezing
    nl = w.clutch["left"].neutral_T[:3, 3]
    assert (np.linalg.norm(c.left_wrist_pose[:3, 3] - nl) < np.linalg.norm(a.left_wrist_pose[:3, 3] - nl))


def test_teledata_is_real_class():
    w, cl, ck = make()
    td = w.get_tele_data()
    assert type(td) is tvw.TeleData
    assert td.head_pose.shape == (4, 4) and td.left_hand_pos.shape == (25, 3)
    assert not td.motion_data_ready


def _ego_R(side):
    z = np.array([0.0, 1.0, 0.0]) if side == "left" else np.array([0.0, -1.0, 0.0])
    x = np.array([1.0, 0.0, 0.0])
    return np.column_stack([x, np.cross(z, x), z])


def test_ego_shared_anchor_keeps_inter_hand_geometry():
    """Lopsided calibration pose; bringing the hands together brings the robot hands together."""
    w, cl, ck = make(mapping="ego")
    cal = {"left": (np.array([0.45, 0.05, -0.35]), _ego_R("left")),
           "right": (np.array([0.40, -0.30, -0.45]), _ego_R("right"))}
    push(cl, ck, cal)
    w.calibrate()
    td = w.get_tele_data()
    dL = td.left_wrist_pose[:3, 3] - td.right_wrist_pose[:3, 3]
    np.testing.assert_allclose(dL, cal["left"][0] - cal["right"][0], atol=1e-9)   # k = 1
    mid = 0.5 * (fallback_neutral()[0][:3, 3] + fallback_neutral()[1][:3, 3])
    np.testing.assert_allclose(0.5 * (td.left_wrist_pose[:3, 3] + td.right_wrist_pose[:3, 3]), mid, atol=1e-9)
    c = np.array([0.45, -0.1, -0.35])
    push(cl, ck, {s: (c, R) for s, (_, R) in cal.items()})
    td = w.get_tele_data()
    np.testing.assert_allclose(td.left_wrist_pose[:3, 3], td.right_wrist_pose[:3, 3], atol=1e-9)


def test_ego_absolute_rotation_neutral_hand_gives_robot_neutral():
    w, cl, ck = make(mapping="ego")
    poses = {"left": (np.array([0.45, 0.15, -0.35]), _ego_R("left")),
             "right": (np.array([0.45, -0.15, -0.35]), _ego_R("right"))}
    push(cl, ck, poses)
    w.calibrate()
    td = w.get_tele_data()
    from xr_camera_teleop.teleop.mano_to_webxr import frame_angle
    # synthetic hand's MCP layout tilts the keypoint frame ~6 deg; real hands differ similarly
    assert frame_angle(td.left_wrist_pose[:3, :3], np.eye(3)) < np.radians(8)
    assert frame_angle(td.right_wrist_pose[:3, :3], np.eye(3)) < np.radians(8)


def test_ego_ignores_far_hands():
    w, cl, ck = make(mapping="ego", max_depth=0.9)
    push(cl, ck, {"left": (np.array([1.3, 0.2, -0.3]), _ego_R("left"))})
    w.get_tele_data()
    assert w._seen_t["left"] is None


def test_auto_yaw_camera_facing_operator():
    """Camera across the table facing the operator: hands point toward the camera (-x in raw W),
    operator's left hand appears on the camera's right (-y). Calibration must recover yaw = 180 deg
    so that left stays left and moving toward the camera is robot-forward."""
    Rz = np.diag([-1.0, -1.0, 1.0])
    w, cl, ck = make(mapping="ego")
    raw = {"left": (np.array([0.5, -0.15, -0.3]), Rz @ syn.rest_rotation("left")),
           "right": (np.array([0.5, 0.15, -0.3]), Rz @ syn.rest_rotation("right"))}
    push(cl, ck, raw)
    w.calibrate()
    td0 = w.get_tele_data()
    assert abs(abs(np.degrees(w.yaw)) - 180) < 10
    assert td0.left_wrist_pose[1, 3] > td0.right_wrist_pose[1, 3]          # left is +y
    moved = {s: (p + np.array([-0.1, 0, 0]), R) for s, (p, R) in raw.items()}   # toward the camera
    push(cl, ck, moved)
    td1 = w.get_tele_data()
    assert td1.left_wrist_pose[0, 3] - td0.left_wrist_pose[0, 3] > 0.08    # robot forward


def test_auto_yaw_egocentric_is_zero():
    w, cl, ck = make(mapping="ego")
    push(cl, ck, rest())
    w.calibrate()
    w.get_tele_data()
    assert abs(np.degrees(w.yaw)) < 10


def _run(w, cl, ck, poses, n, dt=0.033):
    td = None
    for _ in range(n):
        push(cl, ck, poses, dt=dt)
        td = w.get_tele_data()
    return td


def _swap(poses):
    return {"left": poses["right"], "right": poses["left"]}


def test_swapped_labels_return_robot_to_neutral_and_resume():
    w, cl, ck = make(mapping="ego")
    good = rest()
    push(cl, ck, good)
    w.calibrate()
    w.get_tele_data()
    moved = {s: (p + np.array([0.15, 0, 0.1]), R) for s, (p, R) in good.items()}
    td = _run(w, cl, ck, moved, 5)
    n = fallback_neutral()
    assert np.linalg.norm(td.left_wrist_pose[:3, 3] - n[0][:3, 3]) > 0.1
    bad = {s: (p, R) for s, (p, R) in _swap(moved).items()}
    _run(w, cl, ck, bad, 2)                       # shorter than swap_hold: still following
    assert not w.swapped
    td = _run(w, cl, ck, bad, 200)                # ~6.6 s at the clutch return speed
    assert w.swapped and w.status()["swapped"]
    np.testing.assert_allclose(td.left_wrist_pose[:3, 3], n[0][:3, 3], atol=1e-6)
    np.testing.assert_allclose(td.right_wrist_pose[:3, 3], n[1][:3, 3], atol=1e-6)
    np.testing.assert_allclose(td.left_hand_pos, w._default_hand_pos("left"))
    td = _run(w, cl, ck, moved, 60)               # sides correct again: follows the hands
    assert not w.swapped
    assert np.linalg.norm(td.left_wrist_pose[:3, 3] - n[0][:3, 3]) > 0.1


def test_crossing_within_margin_and_single_hand_are_not_swaps():
    w, cl, ck = make(mapping="ego")
    push(cl, ck, rest())
    w.calibrate()
    w.get_tele_data()
    R = {s: syn.rest_rotation(s) for s in ("left", "right")}
    close = {"left": (np.array([0.45, -0.01, -0.3]), R["left"]), "right": (np.array([0.45, 0.01, -0.3]), R["right"])}
    _run(w, cl, ck, close, 30)                    # 2 cm crossed < 3 cm margin
    assert not w.swapped
    _run(w, cl, ck, {"left": (np.array([0.45, -0.3, -0.3]), R["left"])}, 30)   # one hand only: no rule
    assert not w.swapped


def test_calibration_postponed_while_swapped():
    w, cl, ck = make(mapping="ego")
    bad = _swap(rest())
    bad = {s: (p, syn.rest_rotation(s)) for s, (p, _) in bad.items()}
    push(cl, ck, bad)
    w.calibrate()
    w.get_tele_data()
    assert not w.clutch["left"].calibrated
    _run(w, cl, ck, rest(), 1)
    assert w.clutch["left"].calibrated and w.clutch["right"].calibrated
