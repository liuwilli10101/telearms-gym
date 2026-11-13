from __future__ import annotations
from abc import ABC
import numpy as np
from typing import Dict
import re
from mojo.elements import Body, Geom
from mojo.elements.consts import GeomType
import string
from pyquaternion import Quaternion
from roboeval.roboeval_env import RoboEvalEnv
from roboeval.const import PRESETS_PATH, HandSide
from roboeval.envs.props.items import GoalPatch, ColoredCube
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class CubeYawRotation45Deg(RoboEvalEnv, MetricRolloutEval):
    """
    Rotate a single colored cube by 45 degrees around the vertical (table-normal) axis
    so it is diamond-oriented, with one corner facing toward the robot and the opposite
    corner facing away.

    Sequential Sub-Goals:
        1. Grasp the cube (stable grasp suitable for twisting about world Z).
        2. Rotate the cube in-place to reach ~45° yaw about world Z.
        3. Release the cube while keeping the achieved yaw stable on the table.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _TARGET_YAW_DEG = 45.0
    _TOL_DEG = 5.0
    _CUBE_Z_ON_TABLE = 0.97

    _INSTRUCTION_VARIANTS = [
        "Rotate the colored cube 45 degrees around the vertical axis so a corner points toward you and the opposite corner points away.",
        "Twist the cube on the table until it is turned 45° in yaw, with its corners facing toward and away from the robot.",
        "Rotate the cube in place by one-eighth turn (45°) around the up axis so it looks like a diamond from above.",
        "Turn the cube 45 degrees about the vertical axis so it is diagonally oriented relative to the table edges.",
    ]

    # ----------------------- Helpers (local, no new assets) -----------------------

    @staticmethod
    def _wrap_deg(angle_deg: float) -> float:
        """Wrap to [-180, 180)."""
        return (angle_deg + 180.0) % 360.0 - 180.0

    @classmethod
    def _ang_diff_deg(cls, a_deg: float, b_deg: float) -> float:
        """Smallest absolute difference between two angles in degrees."""
        return abs(cls._wrap_deg(a_deg - b_deg))

    @staticmethod
    def _yaw_to_quat_xyzw(yaw_deg: float) -> np.ndarray:
        """Create an XYZW quaternion for a pure yaw about world Z."""
        q = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(np.deg2rad(yaw_deg)))  # wxyz internally
        return np.array([q.x, q.y, q.z, q.w], dtype=float)  # xyzw for set_pose

    @staticmethod
    def _quat_xyzw_to_yaw_deg(quat_xyzw: np.ndarray) -> float:
        """Extract yaw (about world Z) in degrees. NOTE: body.get_quaternion() returns
        wxyz, so the input is wxyz (despite the legacy parameter name)."""
        qw, qx, qy, qz = [float(v) for v in quat_xyzw]
        q = Quaternion(qw, qx, qy, qz)  # wxyz
        yaw_rad, _, _ = q.yaw_pitch_roll  # (yaw, pitch, roll), radians
        return float(np.rad2deg(yaw_rad))

    # ----------------------- RoboEval hooks -----------------------

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Assets (MUST use ONLY ColoredCube)
        self.colored_cube_1 = ColoredCube(self._mojo)
        self.colored_cube_1.set_color("blue")  # REQUIRED in _initialize_env

        # Metrics expect a list
        self.cubes = [self.colored_cube_1]

        # ⚠️ CRITICAL: Initialize metrics with FULL signature
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

    def _on_reset(self):
        # 1. Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2. Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # 3. Task instruction variant
        self._task_instruction = str(rng.choice(self._INSTRUCTION_VARIANTS))

        # 4. REQUIRED: set_color() in _on_reset
        self.colored_cube_1.set_color("blue")

        # 5. RANDOM UNSOLVED placement within bounds
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        pos = np.array(
            [
                float(rng.uniform(0.40, 0.70)),
                float(rng.uniform(-0.25, 0.25)),
                float(self._CUBE_Z_ON_TABLE),
            ],
            dtype=float,
        )

        # Random yaw, explicitly avoid being within tolerance of the target (±5°) so _success() is False after reset.
        # Also avoid the equivalent solution at target±180 (not actually within tolerance, but keep margin anyway).
        while True:
            yaw_deg = float(rng.uniform(-180.0, 180.0))
            if self._ang_diff_deg(yaw_deg, float(self._TARGET_YAW_DEG)) > float(self._TOL_DEG) + 1e-6:
                break

        quat = self._yaw_to_quat_xyzw(yaw_deg)
        self.colored_cube_1.set_pose(position=pos, quat=quat)

        # 6. Physics settling
        self._mojo.physics.forward()

        # 7. Initialize subgoals to False
        for idx in range(3):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        cube = self.colored_cube_1

        # Must be released (not held by any gripper)
        held = any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)
        released = not held

        # Ensure cube is on table surface (not floating / dropped to floor)
        on_table = cube.is_colliding(self.table.table_top)
        if hasattr(self, "_floor"):
            if cube.is_colliding(self._floor):
                on_table = False

        # Orientation check: yaw about world Z near 45 deg
        quat_xyzw = np.array(cube.body.get_quaternion(), dtype=float)
        yaw_deg = self._quat_xyzw_to_yaw_deg(quat_xyzw)
        yaw_err = float(self._ang_diff_deg(yaw_deg, self._TARGET_YAW_DEG))
        orientation_ok = yaw_err <= float(self._TOL_DEG)

        # Subgoal tracking (does not change the goal condition definition)
        # 0: grasp achieved
        if held:
            self._metric_stage(0, True)
        # 1: align achieved
        if orientation_ok and on_table:
            self._metric_stage(1, True)
        # 2: release achieved (while maintaining target orientation on table)
        if orientation_ok and on_table and released:
            self._metric_stage(2, True)

        positions_ok = bool(orientation_ok and on_table)
        semantic_ok = True  # No semantic constraints in this task

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "yaw_error_deg": yaw_err,
            },
            pose_error={
                "yaw_deg": yaw_deg,
                "yaw_error_deg": yaw_err,
                "target_yaw_deg": float(self._TARGET_YAW_DEG),
                "tolerance_deg": float(self._TOL_DEG),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})