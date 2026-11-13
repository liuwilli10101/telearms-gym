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

class RotatedAndAlignedCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """Place red and blue side by side with red rotated 45 degrees and blue unrotated.

Success Criteria:
- The red and blue cubes are both on the table, aligned in Y, and separated along X within the side-by-side bounds used by the code.
- The red cube yaw is 45 degrees within tolerance and the blue cube yaw is 0 degrees within tolerance.
- This task's final success does not require a gripper-release check."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Table / cube geometry
    _CUBE_REST_Z = 0.97

    # Goal tolerances (from task spec)
    _YAW_TOL_DEG = 10.0
    _ALIGN_Y_TOL_M = 0.03
    _LEFT_OF_MIN_DX = 0.04
    _LEFT_OF_MAX_DX = 0.12

    # Additional tight pose tolerance for metrics (registry hint)
    _POS_TOL_M = 0.014

    def _quat_xyzw_from_yaw_deg(self, yaw_deg: float) -> np.ndarray:
        """Create an (x,y,z,w) quaternion for a pure yaw rotation about +Z."""
        q = Quaternion(axis=[0.0, 0.0, 1.0], angle=np.deg2rad(yaw_deg))  # w,x,y,z
        return np.array([q.x, q.y, q.z, q.w], dtype=float)

    def _yaw_deg_from_quat_xyzw(self, quat_xyzw: np.ndarray) -> float:
        """
        Extract yaw (deg) from an (x,y,z,w) quaternion using ZYX convention.
        Returns yaw in [-180, 180].
        """
        x, y, z, w = [float(v) for v in quat_xyzw]
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        yaw_deg = np.rad2deg(yaw)
        # wrap to [-180, 180]
        yaw_deg = (yaw_deg + 180.0) % 360.0 - 180.0
        return yaw_deg

    def _ang_diff_deg(self, a_deg: float, b_deg: float) -> float:
        """Smallest absolute angular difference in degrees."""
        d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
        return abs(float(d))

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Only allowed assets: ColoredCube
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

        # Define a deterministic "solved" goal pose (used for metrics + reset)
        self._red_goal_pos = np.array([0.52, 0.00, self._CUBE_REST_Z], dtype=float)
        self._blue_goal_pos = np.array([0.60, 0.00, self._CUBE_REST_Z], dtype=float)
        self._red_goal_quat = self._quat_xyzw_from_yaw_deg(45.0)
        self._blue_goal_quat = self._quat_xyzw_from_yaw_deg(0.0)

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

        # 2. Deterministic RNG (kept for pipeline compatibility)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 3. Randomize objects into UNSOLVED poses within workspace bounds
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = float(self._CUBE_REST_Z)

        # Keep cubes separated to avoid interpenetration/stacking at reset
        min_xy_sep = 0.06

        # Ensure yaw is NOT within tolerance of the goal yaws so _success() is False after reset
        yaw_margin = self._YAW_TOL_DEG + 5.0

        def sample_yaw_away_from(goal_yaw: float) -> float:
            for _ in range(200):
                yaw = float(rng.uniform(-180.0, 180.0))
                if self._ang_diff_deg(yaw, goal_yaw) > yaw_margin:
                    return yaw
            return float((goal_yaw + 90.0 + rng.uniform(-30.0, 30.0) + 180.0) % 360.0 - 180.0)

        # Sample positions (and keep away from exact goal locations a bit)
        for _ in range(500):
            red_pos = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)
            blue_pos = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi), z], dtype=float)

            if np.linalg.norm(red_pos[:2] - blue_pos[:2]) < min_xy_sep:
                continue

            # Avoid starting essentially solved in position (extra safety)
            if np.linalg.norm(red_pos - self._red_goal_pos) < 0.05 and np.linalg.norm(blue_pos - self._blue_goal_pos) < 0.05:
                continue

            break

        red_yaw = sample_yaw_away_from(45.0)
        blue_yaw = sample_yaw_away_from(0.0)
        red_quat = self._quat_xyzw_from_yaw_deg(red_yaw)
        blue_quat = self._quat_xyzw_from_yaw_deg(blue_yaw)

        self.red_cube.set_pose(position=red_pos, quat=red_quat)
        self.blue_cube.set_pose(position=blue_pos, quat=blue_quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False (match provided 11 subgoals)
        for idx in range(11):
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

        # --- Goal condition checks (positional/rotational only; no grasp-state checks) ---

        # 1) Both cubes on the table surface (collision with table top)
        red_on_table = bool(self.red_cube.is_colliding(self.table.table_top))
        blue_on_table = bool(self.blue_cube.is_colliding(self.table.table_top))

        # 2) Yaw orientation constraints
        red_q = self.red_cube.body.get_quaternion()
        blue_q = self.blue_cube.body.get_quaternion()
        red_yaw = self._yaw_deg_from_quat_xyzw(red_q)
        blue_yaw = self._yaw_deg_from_quat_xyzw(blue_q)

        red_yaw_ok = self._ang_diff_deg(red_yaw, 45.0) <= self._YAW_TOL_DEG
        blue_yaw_ok = self._ang_diff_deg(blue_yaw, 0.0) <= self._YAW_TOL_DEG

        # 3) Side-by-side relation: red left of blue (implemented as smaller X with bounded gap)
        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()

        dx = float(blue_pos[0] - red_pos[0])
        left_of_ok = (dx >= self._LEFT_OF_MIN_DX) and (dx <= self._LEFT_OF_MAX_DX)

        # 4) "Aligned" along Y axis (similar y-position)
        aligned_y_ok = abs(float(red_pos[1] - blue_pos[1])) <= self._ALIGN_Y_TOL_M

        positions_ok = bool(red_on_table and blue_on_table and left_of_ok and aligned_y_ok)
        orientations_ok = bool(red_yaw_ok and blue_yaw_ok)
        all_ok = bool(positions_ok and orientations_ok)

        # --- Metric stage updates (11 subgoals; use spatial proxies only) ---
        red_done = bool(red_on_table and red_yaw_ok)
        blue_done = bool(blue_on_table and blue_yaw_ok)
        pair_done = bool(left_of_ok and aligned_y_ok)

        # Subgoals 1-5 (red)
        for idx in range(0, 5):
            self._metric_stage(idx, red_done)
        # Subgoals 6-10 (blue)
        for idx in range(5, 10):
            self._metric_stage(idx, blue_done)
        # Subgoal 11 (final arrangement)
        self._metric_stage(10, pair_done)

        self._success_check = all_ok

        # --- Finalize metrics (must be last metric call before return) ---
        red_pos_err = float(np.linalg.norm(red_pos - self._red_goal_pos))
        blue_pos_err = float(np.linalg.norm(blue_pos - self._blue_goal_pos))
        red_yaw_err = float(self._ang_diff_deg(red_yaw, 45.0))
        blue_yaw_err = float(self._ang_diff_deg(blue_yaw, 0.0))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_goal_pos": red_pos_err,
                "blue_to_goal_pos": blue_pos_err,
                "pair_dx_error": float(abs(dx - float(self._blue_goal_pos[0] - self._red_goal_pos[0]))),
                "pair_dy": float(abs(float(red_pos[1] - blue_pos[1]))),
            },
            pose_error={
                "red_yaw_error_deg": red_yaw_err,
                "blue_yaw_error_deg": blue_yaw_err,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})