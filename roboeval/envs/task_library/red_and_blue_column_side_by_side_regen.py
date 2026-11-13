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

class AdjacentColorColumns(RoboEvalEnv, MetricRolloutEval):
    """
    Build two adjacent vertical 2-cube columns: a red-on-red stack and a blue-on-blue stack.
    The red column must be to the left (+Y) of the blue column (right, -Y), with both columns
    aligned in X (front/back).

    Sequential Sub-Goals:
        1. Stack a red cube on a red cube (form the red 2-cube column).
        2. Stack a blue cube on a blue cube (form the blue 2-cube column).
        3. Arrange columns so red is left (+Y) of blue (-Y) and aligned in X.
        4. Release all cubes (no cube held in any gripper).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task tolerances (from spec)
    _STACK_XY_TOL = 0.03
    _STACK_DZ_TOL = 0.02
    _ALIGN_X_TOL = 0.03
    _MIN_Y_SEP = 0.04

    _CUBE_SIZE = 0.04
    _Z0 = 0.97
    _DZ = 0.04

    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create exactly four ColoredCubes (and ONLY this asset type)
        self.red_cube_bottom = ColoredCube(self._mojo)
        self.red_cube_top = ColoredCube(self._mojo)
        self.blue_cube_bottom = ColoredCube(self._mojo)
        self.blue_cube_top = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube_bottom.set_color("red")
        self.red_cube_top.set_color("red")
        self.blue_cube_bottom.set_color("blue")
        self.blue_cube_top.set_color("blue")

        self.cubes = [
            self.red_cube_bottom,
            self.red_cube_top,
            self.blue_cube_bottom,
            self.blue_cube_top,
        ]

        # Goal poses (used for metrics / pose error)
        x = 0.55
        y_red = 0.10   # left is +Y
        y_blue = -0.10  # right is -Y
        q = np.array([0.0, 0.0, 0.0, 1.0])

        self._goal_pose = {
            "red_cube_bottom": (np.array([x, y_red, self._Z0]), q),
            "red_cube_top": (np.array([x, y_red, self._Z0 + self._DZ]), q),
            "blue_cube_bottom": (np.array([x, y_blue, self._Z0]), q),
            "blue_cube_top": (np.array([x, y_blue, self._Z0 + self._DZ]), q),
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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (pipeline may add its own later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.red_cube_bottom.set_color("red")
        self.red_cube_top.set_color("red")
        self.blue_cube_bottom.set_color("blue")
        self.blue_cube_top.set_color("blue")

        # 3. Place objects in RANDOM unsolved positions (within workspace bounds)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = self._Z0

        # Keep cubes separated to avoid interpenetration at reset
        min_xy_sep = 1.25 * self._CUBE_SIZE  # > cube size

        def rand_quat_yaw():
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        def sample_xy_points(n=4, max_tries=2000):
            pts = []
            for _ in range(max_tries):
                if len(pts) >= n:
                    break
                p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
                if all(np.linalg.norm(p - q) >= min_xy_sep for q in pts):
                    pts.append(p)
            if len(pts) < n:
                # Fallback to a simple deterministic spread inside bounds
                xs = np.linspace(x_min + 0.05, x_max - 0.05, n)
                ys = np.linspace(y_min + 0.10, y_max - 0.10, n)
                pts = [np.array([xs[i], ys[n - 1 - i]], dtype=float) for i in range(n)]
            return pts

        # Rejection sample until _success() is False after placement
        # (also ensures we don't accidentally start in the solved goal configuration)
        for _ in range(200):
            pts = sample_xy_points(4)
            qs = [rand_quat_yaw() for _ in range(4)]

            # All cubes on the table (not stacked)
            self.red_cube_bottom.set_pose(position=np.array([pts[0][0], pts[0][1], z0]), quat=qs[0])
            self.red_cube_top.set_pose(position=np.array([pts[1][0], pts[1][1], z0]), quat=qs[1])
            self.blue_cube_bottom.set_pose(position=np.array([pts[2][0], pts[2][1], z0]), quat=qs[2])
            self.blue_cube_top.set_pose(position=np.array([pts[3][0], pts[3][1], z0]), quat=qs[3])

            self._mojo.physics.forward()

            if not self._success():
                break

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _xy_dist(self, a, b) -> float:
        pa = a.body.get_position()
        pb = b.body.get_position()
        return float(np.linalg.norm((pa - pb)[:2]))

    def _is_stacked_on(self, top, bottom) -> bool:
        pt = top.body.get_position()
        pb = bottom.body.get_position()

        xy_ok = self._xy_dist(top, bottom) <= self._STACK_XY_TOL
        dz = float(pt[2] - pb[2])
        z_ok = abs(dz - self._DZ) <= self._STACK_DZ_TOL

        # Collision-based sanity: top should contact bottom, and should not be on the table
        coll_ok = top.is_colliding(bottom)
        not_on_table = not top.is_colliding(self.table.table_top)

        return bool(xy_ok and z_ok and coll_ok and not_on_table)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Color verification (ColoredCube content)
        colors_ok = (
            self.red_cube_bottom.get_color() == "red"
            and self.red_cube_top.get_color() == "red"
            and self.blue_cube_bottom.get_color() == "blue"
            and self.blue_cube_top.get_color() == "blue"
        )

        red_stacked = self._is_stacked_on(self.red_cube_top, self.red_cube_bottom)
        blue_stacked = self._is_stacked_on(self.blue_cube_top, self.blue_cube_bottom)

        prb = self.red_cube_bottom.body.get_position()
        pbb = self.blue_cube_bottom.body.get_position()

        # Left/right uses Y axis: left is +Y (higher y)
        left_right_ok = float(prb[1] - pbb[1]) >= self._MIN_Y_SEP

        # Aligned in X (front/back)
        x_aligned_ok = abs(float(prb[0] - pbb[0])) <= self._ALIGN_X_TOL

        positions_ok = bool(red_stacked and blue_stacked and left_right_ok and x_aligned_ok)

        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging
        if red_stacked and colors_ok:
            self._metric_stage(0, True)
        if blue_stacked and colors_ok:
            self._metric_stage(1, True)
        if (red_stacked and blue_stacked and left_right_ok and x_aligned_ok and colors_ok):
            self._metric_stage(2, True)
        if released:
            self._metric_stage(3, True)

        self._success_check = bool(positions_ok and colors_ok and released)

        # Metrics: distances / pose errors vs the goal
        def pose_err(obj, goal_pos):
            p = obj.body.get_position()
            return float(np.linalg.norm(p - goal_pos))

        red_on_red_xy = self._xy_dist(self.red_cube_top, self.red_cube_bottom)
        blue_on_blue_xy = self._xy_dist(self.blue_cube_top, self.blue_cube_bottom)
        y_sep = float(prb[1] - pbb[1])
        x_misalign = abs(float(prb[0] - pbb[0]))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_stack_xy": red_on_red_xy,
                "blue_stack_xy": blue_on_blue_xy,
                "y_separation(red_minus_blue)": y_sep,
                "x_misalignment": x_misalign,
            },
            pose_error={
                "red_cube_bottom": pose_err(self.red_cube_bottom, self._goal_pose["red_cube_bottom"][0]),
                "red_cube_top": pose_err(self.red_cube_top, self._goal_pose["red_cube_top"][0]),
                "blue_cube_bottom": pose_err(self.blue_cube_bottom, self._goal_pose["blue_cube_bottom"][0]),
                "blue_cube_top": pose_err(self.blue_cube_top, self._goal_pose["blue_cube_top"][0]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})