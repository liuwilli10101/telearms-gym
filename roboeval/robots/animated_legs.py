"""Animated legs for the floating base mode."""

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from dm_control import mjcf
from mojo import Mojo
from mojo.elements import Body, Site, MujocoElement
from pyquaternion import Quaternion

from roboeval.const import ASSETS_PATH, HandSide


class AnimatedLegs(ABC):
    """Animated legs base class."""

    def __init__(self, mojo: Mojo, pelvis: Body):
        """Init."""
        self._mojo = mojo
        self._pelvis = pelvis

    @abstractmethod
    def step(self, pelvis_z: float, is_moving: bool = True):
        """Step animation."""
        ...


class H1AnimatedLegs(AnimatedLegs):
    """Animated legs for H1."""

    MODEL = ASSETS_PATH / "h1/h1_floating_base.xml"
    TORSO_JOINT = "h1_floating_base"

    LEG_LINKS = {
        HandSide.LEFT: [
            "left_hip_yaw_link",
            "left_hip_roll_link",
            "left_hip_pitch_link",
            "left_knee_link",
            "left_ankle_link",
        ],
        HandSide.RIGHT: [
            "right_hip_yaw_link",
            "right_hip_roll_link",
            "right_hip_pitch_link",
            "right_knee_link",
            "right_ankle_link",
        ],
    }

    _LINKS_COUNT = 3
    _L1 = 0.4
    _L2 = 0.4
    _2_L1_L2 = 2 * _L1 * _L2
    _L1SQ_L2SQ = _L1 * _L1 + _L2 * _L2

    _HIPS_OFFSET = 0.1742
    _ANKLE_HEIGHT = 0.08

    _STEP_HEIGHT = 0.04
    _STEP_DURATION = 0.5

    def __init__(self, mojo: Mojo, pelvis: Body):
        """Init."""
        super().__init__(mojo, pelvis)

        # Remove original legs and pelvis mesh
        for side in HandSide:
            Body.get(self._mojo, self.LEG_LINKS[side][0], pelvis).mjcf.remove()
        self._pelvis.geoms[0].mjcf.remove()

        floating_base_site = Site.create(
            self._mojo, self._pelvis, size=np.array([0.001, 0.001, 0.001])
        )
        self._mojo.load_model(self.MODEL, floating_base_site, on_loaded=self._on_loaded)

    def step(self, pelvis_z: float, is_moving: bool = True):
        """Step animation."""
        scale = 1 if is_moving else 0
        solution_collision_min = self._solve(pelvis_z, 0)
        solution_collision_max = self._solve(pelvis_z, self._STEP_HEIGHT * scale)
        solutions_visual = {
            HandSide.LEFT: self._solve(pelvis_z, self._get_offset(0, scale)),
            HandSide.RIGHT: self._solve(pelvis_z, self._get_offset(np.pi / 2, scale)),
        }
        for side in HandSide:
            self._set_leg_state(self._visual[side], solutions_visual[side])
            self._set_leg_state(self._collision_min[side], solution_collision_min)
            self._set_leg_state(self._collision_max[side], solution_collision_max)

    def _on_loaded(self, model: mjcf.RootElement):
        base = MujocoElement(self._mojo, model)

        self._visual: dict[HandSide, list[Body]] = defaultdict(list)
        self._collision_min: dict[HandSide, list[Body]] = defaultdict(list)
        self._collision_max: dict[HandSide, list[Body]] = defaultdict(list)

        for side in HandSide:
            for link in self.LEG_LINKS[side][-self._LINKS_COUNT :]:
                self._visual[side].append(Body.get(self._mojo, link, base))
                self._collision_min[side].append(
                    Body.get(self._mojo, f"{link}_collision_min", base)
                )
                self._collision_max[side].append(
                    Body.get(self._mojo, f"{link}_collision_max", base)
                )

    def _get_offset(self, shift: float, scale: float) -> float:
        scale = np.clip(scale, 0, 1)
        t = self._mojo.physics.time() % self._STEP_DURATION
        return (
            scale
            * self._STEP_HEIGHT
            * (1 - np.abs(np.sin(2 * np.pi * (t / self._STEP_DURATION) + shift)))
        )

    def _set_leg_state(self, leg: list[Body], state: np.ndarray):
        for angle, link in zip(state, leg):
            link = self._mojo.physics.bind(link.mjcf)
            quat = Quaternion(axis=[0, 1, 0], radians=angle)
            link.quat = quat.elements

    def _solve(self, pelvis_z: float, offset: float = 0.0) -> np.ndarray:
        hip_z = pelvis_z - self._HIPS_OFFSET
        r = hip_z - self._ANKLE_HEIGHT - offset
        r = np.clip(r, 0, self._L1 + self._L2)
        angle_knee = np.arccos((self._L1SQ_L2SQ - r * r) / self._2_L1_L2)
        angle_hip = -(np.pi - angle_knee) / 2
        angle_knee = np.pi - angle_knee
        angle_ankle = angle_hip
        return np.array([angle_hip, angle_knee, angle_ankle])
