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

class CubePyramidBuilding(RoboEvalEnv, MetricRolloutEval):
    """
    Build a small 4-cube pyramid: arrange three cubes in a left-to-right base row,
    then stack a fourth cube on top of the middle base cube.

    Coordinate conventions (CRITICAL):
      - X: front/back
      - Y: left/right  (LEFT = +Y, RIGHT = -Y)
      - Z: up

    Sequential Sub-Goals (single-arm executed sequentially):
        1. Place a cube at the left base position (+Y).
        2. Place a cube at the middle base position (Y≈0).
        3. Place a cube at the right base position (-Y).
        4. Stack the final cube on top of the middle base cube.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # ---- Tunables (match task spec tolerances) ----
    _X_ALIGN_TOL_M = 0.03
    _Z_TABLE_TOL_M = 0.02
    _MIN_Y_DELTA_M = 0.03
    _STACK_XY_TOL_M = 0.03
    _STACK_DZ_TARGET_M = 0.04
    _STACK_DZ_TOL_M = 0.02

    # Workspace bounds (must remain within)
    _X_BOUNDS = (0.35, 0.75)
    _Y_BOUNDS = (-0.30, 0.30)

    # ColoredCube geometry
    _Z_BASE = 0.97
    _Z_STACKED = 1.01  # 0.97 + 0.04

    # Metrics stages: keep a fixed count
    _NUM_SUBGOALS = 11

    def _initialize_env(self):
        # Table (already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Objects: MUST use ONLY ColoredCube ---
        self.base_left_cube = ColoredCube(self._mojo)
        self.base_middle_cube = ColoredCube(self._mojo)
        self.base_right_cube = ColoredCube(self._mojo)
        self.top_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env
        self.base_left_cube.set_color("red")
        self.base_middle_cube.set_color("green")
        self.base_right_cube.set_color("blue")
        self.top_cube.set_color("yellow")

        self.cubes = [
            self.base_left_cube,
            self.base_middle_cube,
            self.base_right_cube,
            self.top_cube,
        ]

        # Goal positions (Y axis is left/right; +Y is left)
        center_x = 0.55
        center_y = 0.0
        spacing_y = 0.06  # >= 0.05 suggested spacing

        self._goal_pos = {
            "base_left_cube": np.array([center_x, center_y + spacing_y, self._Z_BASE]),
            "base_middle_cube": np.array([center_x, center_y, self._Z_BASE]),
            "base_right_cube": np.array([center_x, center_y - spacing_y, self._Z_BASE]),
            "top_cube": np.array([center_x, center_y, self._Z_STACKED]),
        }

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

        # 2. Deterministic RNG (pipeline may randomize later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset
        self.base_left_cube.set_color("red")
        self.base_middle_cube.set_color("green")
        self.base_right_cube.set_color("blue")
        self.top_cube.set_color("yellow")

        # 3. Place objects in RANDOM *unsolved* positions on the table (not stacked).
        #    Keep within required workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._Z_BASE

        # Sample non-overlapping positions (simple rejection sampling)
        min_xy_sep = 0.075
        positions = []
        for _ in range(2000):
            if len(positions) == 4:
                break
            x = float(rng.uniform(x_min, x_max))
            y = float(rng.uniform(y_min, y_max))
            p = np.array([x, y, z], dtype=float)
            if all(np.linalg.norm(p[:2] - q[:2]) >= min_xy_sep for q in positions):
                positions.append(p)
        if len(positions) < 4:
            # Fallback (still within bounds) with guaranteed separation by construction
            xs = np.linspace(x_min + 0.05, x_max - 0.05, 4)
            ys = np.array([0.20, 0.05, -0.10, -0.23])
            positions = [np.array([float(xs[i]), float(ys[i]), z], dtype=float) for i in range(4)]

        # Random yaw orientations (still stable on table)
        def yaw_quat(yaw: float) -> np.ndarray:
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=float(yaw))
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        quats = [yaw_quat(rng.uniform(-np.pi, np.pi)) for _ in range(4)]

        # Place all cubes on the table (including "top_cube") so stacking is impossible -> _success() must be False
        self.base_left_cube.set_pose(position=positions[0], quat=quats[0])
        self.base_middle_cube.set_pose(position=positions[1], quat=quats[1])
        self.base_right_cube.set_pose(position=positions[2], quat=quats[2])
        self.top_cube.set_pose(position=positions[3], quat=quats[3])
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    # ----------------- Helper checks -----------------
    def _in_bounds(self, p: np.ndarray) -> bool:
        return (self._X_BOUNDS[0] <= float(p[0]) <= self._X_BOUNDS[1]) and (
            self._Y_BOUNDS[0] <= float(p[1]) <= self._Y_BOUNDS[1]
        )

    def _on_table(self, cube: ColoredCube) -> bool:
        p = cube.body.get_position()
        # Collision with table top is the robust indicator of being on table
        return bool(cube.is_colliding(self.table.table_top) and abs(float(p[2]) - self._Z_BASE) <= self._Z_TABLE_TOL_M)

    def _x_aligned(self, cubes) -> bool:
        xs = [float(c.body.get_position()[0]) for c in cubes]
        x_mean = float(np.mean(xs))
        return all(abs(x - x_mean) <= self._X_ALIGN_TOL_M for x in xs)

    def _stacked_on(self, top: ColoredCube, base: ColoredCube) -> bool:
        pt = top.body.get_position()
        pb = base.body.get_position()

        if not top.is_colliding(base):
            return False
        if top.is_colliding(self.table.table_top):
            return False

        dxy = float(np.linalg.norm(np.array([pt[0] - pb[0], pt[1] - pb[1]])))
        dz = float(pt[2] - pb[2])
        return (dxy <= self._STACK_XY_TOL_M) and (abs(dz - self._STACK_DZ_TARGET_M) <= self._STACK_DZ_TOL_M) and (dz > 0.0)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_pos"):
            return False

        # Workspace bounds sanity
        if not all(self._in_bounds(c.body.get_position()) for c in self.cubes):
            return False

        # --- Base row checks ---
        left_p = self.base_left_cube.body.get_position()
        mid_p = self.base_middle_cube.body.get_position()
        right_p = self.base_right_cube.body.get_position()

        base_on_table = self._on_table(self.base_left_cube) and self._on_table(self.base_middle_cube) and self._on_table(self.base_right_cube)

        # Left-to-right along Y: leftmost has highest Y, rightmost has lowest Y
        y_order_ok = (float(left_p[1]) - float(mid_p[1]) >= self._MIN_Y_DELTA_M) and (
            float(mid_p[1]) - float(right_p[1]) >= self._MIN_Y_DELTA_M
        )

        x_align_ok = self._x_aligned([self.base_left_cube, self.base_middle_cube, self.base_right_cube])

        # --- Top stack check ---
        stacked_ok = self._stacked_on(self.top_cube, self.base_middle_cube)

        positions_ok = bool(base_on_table and y_order_ok and x_align_ok and stacked_ok)

        # Released gripper condition (generic across robots)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g) for obj in self.cubes for g in self.robot.grippers
        )

        # Update stage metrics (lightweight, geometry-based)
        if self._on_table(self.base_left_cube):
            self._metric_stage(0, True)
        if self._on_table(self.base_right_cube):
            self._metric_stage(1, True)
        if self._on_table(self.base_middle_cube):
            self._metric_stage(2, True)
        if base_on_table and y_order_ok:
            self._metric_stage(3, True)
        if base_on_table and x_align_ok:
            self._metric_stage(4, True)
        if stacked_ok:
            self._metric_stage(5, True)
        if released:
            self._metric_stage(6, True)

        # If fully correct, mark remaining stages as achieved (task-level completion)
        if positions_ok and released:
            for idx in range(self._NUM_SUBGOALS):
                self._metric_stage(idx, True)

        self._success_check = bool(positions_ok and released)

        # Metrics: distances to goal positions
        def pos_err(cube: ColoredCube, goal: np.ndarray) -> float:
            p = cube.body.get_position()
            return float(np.linalg.norm(np.array(p) - np.array(goal)))

        target_distance = {
            "base_left_cube": pos_err(self.base_left_cube, self._goal_pos["base_left_cube"]),
            "base_middle_cube": pos_err(self.base_middle_cube, self._goal_pos["base_middle_cube"]),
            "base_right_cube": pos_err(self.base_right_cube, self._goal_pos["base_right_cube"]),
            "top_cube": pos_err(self.top_cube, self._goal_pos["top_cube"]),
        }
        pose_error = dict(target_distance)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})