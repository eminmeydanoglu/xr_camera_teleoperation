"""Programmatic MJCF for floating human hands + helpers to pose them each frame."""
from __future__ import annotations

import mujoco
import numpy as np

from .mapping import BONES, HANDS

COLORS = {  # live / stale(grey)
    "left": {"live": (0.25, 0.85, 0.35, 1.0), "bone": (0.35, 0.95, 0.45, 1.0)},
    "right": {"live": (0.25, 0.5, 1.0, 1.0), "bone": (0.4, 0.65, 1.0, 1.0)},
}
GREY = (0.55, 0.55, 0.55, 0.6)
HIDDEN_POS = np.array([0.0, 0.0, -50.0])
JOINT_R, BONE_R, AXIS_LEN, AXIS_R = 0.008, 0.0045, 0.06, 0.0025
AXES = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # RGB = xyz


def _body(name, geom, rgba):
    return (f'<body name="{name}" mocap="true" pos="0 0 -50">'
            f'<geom name="g_{name}" {geom} rgba="{" ".join(map(str, rgba))}" '
            f'contype="0" conaffinity="0"/></body>')


def build_mjcf(cam_height=1.5, waist_height=None) -> str:
    bodies = []
    for h in HANDS:
        for i in range(21):
            r = JOINT_R * (1.5 if i == 0 else 1.0)
            bodies.append(_body(f"{h}_j{i}", f'type="sphere" size="{r}"', COLORS[h]["live"]))
        for b in range(20):
            bodies.append(_body(f"{h}_b{b}", f'type="capsule" size="{BONE_R} 0.01"', COLORS[h]["bone"]))
        for a, c in zip("xyz", AXES):
            bodies.append(_body(f"{h}_ax{a}", f'type="capsule" size="{AXIS_R} {AXIS_LEN/2}"', c))
    waist = ""
    if waist_height is not None:  # robot-waist reference: frame triad + box at the pelvis origin
        waist = (f'<body name="waist" pos="0 0 {waist_height}">'
                 '<geom type="box" size="0.12 0.18 0.03" rgba="0.9 0.7 0.2 0.35" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.005" rgba="1 0 0 1" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" fromto="0 0 0 0 0.15 0" size="0.005" rgba="0 1 0 1" contype="0" conaffinity="0"/>'
                 '<geom type="capsule" fromto="0 0 0 0 0 0.15" size="0.005" rgba="0 0 1 1" contype="0" conaffinity="0"/>'
                 '</body>')
    return f"""
<mujoco model="floating_hands">
  <visual><global offwidth="1280" offheight="720"/><headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6"/>
    <quality shadowsize="2048"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.6" rgb2="0.05 0.05 0.1" width="64" height="64"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.24 0.28" rgb2="0.17 0.19 0.22" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="1 0 4" dir="0 0 -1" diffuse="0.7 0.7 0.7"/>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid" contype="0" conaffinity="0"/>
    <body name="head" pos="0 0 {cam_height}">
      <geom type="sphere" size="0.07" rgba="0.8 0.8 0.8 0.3" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.1 0 0" size="0.04 0.03 0.03" rgba="1 0.8 0.2 0.8" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0.1 0" size="0.02 0.01 0.02" rgba="0.3 1 0.4 0.8" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 -0.1 0" size="0.02 0.01 0.02" rgba="0.4 0.6 1 0.8" contype="0" conaffinity="0"/>
    </body>
    <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.006" rgba="1 0 0 1" contype="0" conaffinity="0"/>
    <geom type="capsule" fromto="0 0 0 0 0.3 0" size="0.006" rgba="0 1 0 1" contype="0" conaffinity="0"/>
    <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.006" rgba="0 0 1 1" contype="0" conaffinity="0"/>
    {waist}
    {"".join(bodies)}
  </worldbody>
</mujoco>"""


class HandsScene:
    def __init__(self, cam_height=1.5, waist_height=None):
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(cam_height, waist_height))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.mocap = {}
        self.geom = {}
        for h in HANDS:
            names = ([f"{h}_j{i}" for i in range(21)] + [f"{h}_b{b}" for b in range(20)]
                     + [f"{h}_ax{a}" for a in "xyz"])
            self.mocap[h] = [m.body_mocapid[m.body(n).id] for n in names]
            self.geom[h] = [m.geom(f"g_{n}").id for n in names]
        m.geom_rbound[:] = np.maximum(m.geom_rbound, 0.25)
        self._state = {h: None for h in HANDS}
        self._live_rgba = m.geom_rgba.copy()

    def set_state(self, h, state):
        """state: 'live' | 'stale' | 'hidden'. Recolors on change only."""
        if self._state[h] == state:
            return
        self._state[h] = state
        ids = self.geom[h]
        n = 41  # joints+bones; triad (last 3) keeps RGB
        for k, g in enumerate(ids[:n]):
            self.model.geom_rgba[g] = GREY if state == "stale" else self._live_rgba[g]
        if state == "hidden":
            for mid in self.mocap[h]:
                self.data.mocap_pos[mid] = HIDDEN_POS

    def pose(self, h, wrist_p, wrist_R, kps):
        d = self.data
        ids = self.mocap[h]
        for i in range(21):
            d.mocap_pos[ids[i]] = kps[i]
        q = np.zeros(4)
        for b, (a, c) in enumerate(BONES):
            p0, p1 = kps[a], kps[c]
            v = p1 - p0
            L = np.linalg.norm(v)
            mid = ids[21 + b]
            d.mocap_pos[mid] = 0.5 * (p0 + p1)
            if L < 1e-6:
                continue
            mujoco.mju_quatZ2Vec(q, v / L)
            d.mocap_quat[mid] = q
            self.model.geom_size[self.geom[h][21 + b]][1] = 0.5 * L
        qq = np.zeros(4)
        for k in range(3):
            mid = ids[41 + k]
            ax = wrist_R[:, k]
            d.mocap_pos[mid] = wrist_p + 0.5 * AXIS_LEN * ax
            mujoco.mju_quatZ2Vec(q, ax)
            d.mocap_quat[mid] = q

    def forward(self):
        mujoco.mj_forward(self.model, self.data)


def default_camera(cam_height=1.5):
    """(lookat, distance, azimuth, elevation) for a camera behind & above the operator's head."""
    return np.array([0.3, 0.0, cam_height - 0.05]), 1.1, 30.0, -20.0
