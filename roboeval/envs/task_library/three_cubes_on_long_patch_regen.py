# Standard imports that are ALWAYS prepended to task code
# This is the canonical source - validation agents should strip imports from LLM output

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
from roboeval.envs.props.items import ColoredCube, GoalPatch
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class CubeLinePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes onto a long rectangular green goal patch and arrange them
    in a straight line along the +Y (leftward) axis. The goal patch must be oriented
    with its long axis aligned to +Y.

    Sequential Sub-Goals:
        1. Grasp cube 1 with either gripper.
        2. Move cube 1 onto the green goal patch.
        3. Place and release cube 1 on the goal patch.
        4. Grasp cube 2 with either gripper.
        5. Move cube 2 onto the green goal patch.
        6. Place and release cube 2 on the goal patch.
        7. Grasp cube 3 with either gripper.
        8. Move cube 3 onto the green goal patch.
        9. Place and release cube 3 on the goal patch.
        10. Adjust cube positions so they form a straight line along +Y while keeping X nearly constant.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace / geometry constants
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)

    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    # Long axis must be +Y, so make patch "depth" (local Y) longer than "width" (local X).
    _GOAL_PATCH_SIZE = (0.06, 0.22)  # (x, y) in meters

    # Goal tolerances from spec
    _PATCH_YAW_TOL_RAD = 0.15
    _MAX_ABS_X_DEV_M = 0.03
    _MIN_ADJ_SPACING_M = 0.04
    _MAX_ADJ_SPACING_M = 0.12

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # --- Objects (ONLY allowed assets) ---
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cube_3 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2, self.cube_3]

        # REQUIRED: set_color() in BOTH _initialize_env AND _on_reset
        self._cube_colors = ["red", "blue", "yellow"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        self.goal_patch_green = GoalPatch(
            self._mojo, color="green", size=self._GOAL_PATCH_SIZE, name="goal_patch_green"
        )

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

    def _quat_from_yaw(self, yaw_rad: float) -> np.ndarray:
        """Quaternion (x,y,z,w) for rotation about world +Z by yaw."""
        return np.array([0.0, 0.0, float(np.sin(yaw_rad / 2.0)), float(np.cos(yaw_rad / 2.0))])

    def _yaw_from_quat_xyzw(self, q: np.ndarray) -> float:
        """Extract yaw (about Z) from quaternion stored as (x,y,z,w)."""
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _angle_to_plus_y_for_patch_long_axis(self) -> float:
        """
        Compute angle between patch long axis (local +Y because size_y > size_x)
        and world +Y. Direction matters (+Y, not -Y).
        """
        q = self.goal_patch_green.body.get_quaternion()
        yaw = self._yaw_from_quat_xyzw(np.array(q))

        # Local +Y rotated by yaw: (x,y) = (-sin(yaw), cos(yaw))
        long_axis_world = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=float)
        world_plus_y = np.array([0.0, 1.0, 0.0], dtype=float)

        dot = float(np.clip(np.dot(long_axis_world, world_plus_y), -1.0, 1.0))
        return float(np.arccos(dot))

    def _is_block_on_patch(self, block: ColoredCube, patch: GoalPatch) -> bool:
        """
        Reliable "on patch" check using the GoalPatch .collider pattern.
        """
        if not hasattr(patch, "collider"):
            return False
        if not block.is_colliding(patch.collider):
            return False
        # If a floor exists and the block is on the floor, reject (defensive)
        if hasattr(self, "_floor") and block.is_colliding(self._floor):
            return False
        return True

    def _compute_line_along_plus_y_ok(self) -> Dict[str, float | bool]:
        """
        Check line pattern:
          - nearly constant X (index 0)
          - cube_1, cube_2, cube_3 monotonic increasing in Y (index 1), with spacing bounds
        """
        p1 = np.array(self.cube_1.body.get_position(), dtype=float)
        p2 = np.array(self.cube_2.body.get_position(), dtype=float)
        p3 = np.array(self.cube_3.body.get_position(), dtype=float)

        xs = np.array([p1[0], p2[0], p3[0]], dtype=float)
        ys = np.array([p1[1], p2[1], p3[1]], dtype=float)

        x_mean = float(np.mean(xs))
        max_abs_x_dev = float(np.max(np.abs(xs - x_mean)))

        # Require strict monotonic +Y ordering: cube_1 (rightmost) < cube_2 < cube_3 (leftmost)
        ordered_ok = bool((ys[0] < ys[1]) and (ys[1] < ys[2]))

        d12 = float(ys[1] - ys[0])
        d23 = float(ys[2] - ys[1])
        spacing_ok = bool(
            (self._MIN_ADJ_SPACING_M <= d12 <= self._MAX_ADJ_SPACING_M)
            and (self._MIN_ADJ_SPACING_M <= d23 <= self._MAX_ADJ_SPACING_M)
        )

        line_ok = bool((max_abs_x_dev <= self._MAX_ABS_X_DEV_M) and ordered_ok and spacing_ok)

        return {
            "line_ok": line_ok,
            "max_abs_x_dev": max_abs_x_dev,
            "y_spacing_12": d12,
            "y_spacing_23": d23,
            "ordered_ok": ordered_ok,
            "spacing_ok": spacing_ok,
        }

    def _clamp_xy_to_workspace(self, x: float, y: float) -> tuple[float, float]:
        x = float(np.clip(x, self._X_BOUNDS[0], self._X_BOUNDS[1]))
        y = float(np.clip(y, self._Y_BOUNDS[0], self._Y_BOUNDS[1]))
        return x, y

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env AND _on_reset
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM (unsolved) configuration within required workspace bounds
        X_MIN, X_MAX = 0.40, 0.70
        Y_MIN, Y_MAX = -0.25, 0.25

        def clamp_req(x: float, y: float) -> tuple[float, float]:
            return float(np.clip(x, X_MIN, X_MAX)), float(np.clip(y, Y_MIN, Y_MAX))

        # Patch position (orientation unchanged / identity)
        patch_center_x = float(rng.uniform(X_MIN, X_MAX))
        patch_center_y = float(rng.uniform(Y_MIN, Y_MAX))
        patch_center_x, patch_center_y = clamp_req(patch_center_x, patch_center_y)
        patch_yaw = 0.0
        self.goal_patch_green.body.set_position(np.array([patch_center_x, patch_center_y, self._PATCH_Z], dtype=float))

        # Define a goal pose (for metrics) but do NOT spawn cubes there.
        cube_x_goal = float(patch_center_x + rng.uniform(-0.004, 0.004))
        cube_x_goal, _ = clamp_req(cube_x_goal, patch_center_y)
        y_offsets = [-0.06, 0.00, 0.06]
        y_noise = [float(rng.uniform(-0.002, 0.002)) for _ in range(3)]
        goal_cube_positions = []
        for off, eps in zip(y_offsets, y_noise):
            xg, yg = clamp_req(cube_x_goal, patch_center_y + off + eps)
            goal_cube_positions.append(np.array([xg, yg, self._CUBE_Z], dtype=float))

        self._goal_cube_positions = goal_cube_positions
        self._goal_patch_pose = (
            np.array([patch_center_x, patch_center_y, self._PATCH_Z], dtype=float),
            self._quat_from_yaw(patch_yaw),
        )

        # Sample cube positions that are very unlikely to be on the patch to ensure _success() == False.
        # Success requires ALL cubes to be on the patch, so forcing at least one cube far from the patch suffices.
        def sample_cube_far_from_patch() -> np.ndarray:
            # Patch half-extents: x=0.03, y=0.11. Use a conservative clearance margin.
            for _ in range(200):
                x = float(rng.uniform(X_MIN, X_MAX))
                y = float(rng.uniform(Y_MIN, Y_MAX))
                # Ensure far in at least one axis from patch center.
                if (abs(x - patch_center_x) > 0.08) or (abs(y - patch_center_y) > 0.18):
                    x, y = clamp_req(x, y)
                    return np.array([x, y, self._CUBE_Z], dtype=float)
            # Fallback: force far Y with clipping
            x, y = clamp_req(patch_center_x, patch_center_y + 0.23)
            return np.array([x, y, self._CUBE_Z], dtype=float)

        # Place cubes randomly (unsolved): at least cube_1 is guaranteed far from patch.
        cube_positions = [sample_cube_far_from_patch(), sample_cube_far_from_patch(), sample_cube_far_from_patch()]
        for cube, pos in zip(self.cubes, cube_positions):
            cube.set_pose(position=pos, quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float))

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "goal_patch_green"):
            return False

        # --- Goal checks ---
        patch_align_err = self._angle_to_plus_y_for_patch_long_axis()
        patch_ok = bool(patch_align_err <= self._PATCH_YAW_TOL_RAD)

        on_patch_1 = self._is_block_on_patch(self.cube_1, self.goal_patch_green)
        on_patch_2 = self._is_block_on_patch(self.cube_2, self.goal_patch_green)
        on_patch_3 = self._is_block_on_patch(self.cube_3, self.goal_patch_green)
        on_patch_all = bool(on_patch_1 and on_patch_2 and on_patch_3)

        line_info = self._compute_line_along_plus_y_ok()
        line_ok = bool(line_info["line_ok"])

        positions_ok = bool(patch_ok and on_patch_all and line_ok)
        semantic_ok = True  # spatial-only task

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort observables) ---
        # Mark per-cube grasp/transport/place as achieved once that cube is on the patch and released.
        # (We do not attempt to infer an actual grasp event.)
        if released and on_patch_1:
            for sg in [0, 1, 2]:
                self._metric_stage(sg, True)
        if released and on_patch_2:
            for sg in [3, 4, 5]:
                self._metric_stage(sg, True)
        if released and on_patch_3:
            for sg in [6, 7, 8]:
                self._metric_stage(sg, True)
        if released and patch_ok and line_ok and on_patch_all:
            self._metric_stage(9, True)

        # --- Final success flag ---
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics / errors ---
        pose_error = {"goal_patch_long_axis_to_+y_rad": float(patch_align_err)}
        if hasattr(self, "_goal_cube_positions"):
            target_distance = {
                "cube_1_to_goal_m": float(np.linalg.norm(np.array(self.cube_1.body.get_position()) - self._goal_cube_positions[0])),
                "cube_2_to_goal_m": float(np.linalg.norm(np.array(self.cube_2.body.get_position()) - self._goal_cube_positions[1])),
                "cube_3_to_goal_m": float(np.linalg.norm(np.array(self.cube_3.body.get_position()) - self._goal_cube_positions[2])),
                "line_max_abs_x_dev_m": float(line_info["max_abs_x_dev"]),
            }
        else:
            target_distance = {
                "line_max_abs_x_dev_m": float(line_info["max_abs_x_dev"]),
            }

        pose_error.update(
            {
                "line_y_spacing_12_m": float(line_info["y_spacing_12"]),
                "line_y_spacing_23_m": float(line_info["y_spacing_23"]),
            }
        )

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})