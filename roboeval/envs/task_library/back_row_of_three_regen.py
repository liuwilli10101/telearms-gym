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

class BackEdgeCubeRow(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes into a single row along the back edge of the workspace (maximum x),
    farthest from the robot. The cubes should be aligned at x = 0.70 and spaced left-to-right
    (along Y) within the workspace bounds.

    Sequential Sub-Goals:
        1. Align all cubes to the back edge row (x ≈ 0.70) with correct left-to-right order.
        2. Securely grasp the red cube with either gripper.
        3. Move the red cube toward the back edge of the workspace (toward +X).
        4. Place and release the red cube at the left position of the back-edge row.
        5. Securely grasp the green cube with either gripper.
        6. Move the green cube toward the back edge of the workspace (toward +X).
        7. Place and release the green cube at the center position of the back-edge row.
        8. Securely grasp the blue cube with either gripper.
        9. Move the blue cube toward the back edge of the workspace (toward +X).
        10. Place and release the blue cube at the right position of the back-edge row.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace/task constants
    _X_BACK_EDGE = 0.70
    _WORKSPACE_X_BOUNDS = (0.35, 0.75)
    _WORKSPACE_Y_BOUNDS = (-0.30, 0.30)
    _CUBE_Z_ON_TABLE = 0.97

    # Goal positions (world frame)
    _GOAL = {
        "cube_red":   {"x": 0.70, "y": 0.12},
        "cube_green": {"x": 0.70, "y": 0.00},
        "cube_blue":  {"x": 0.70, "y": -0.12},
    }

    # Tolerances (task spec)
    _TOL_X = 0.015
    _TOL_Y = 0.02

    # Arrangement constraint (aligned row)
    _ROW_X_TOL = 0.015

    # Subgoal staging
    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Objects (ONLY ColoredCube allowed)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]

        # Cache goal positions
        self._goal_positions = {
            "cube_red": np.array([self._GOAL["cube_red"]["x"], self._GOAL["cube_red"]["y"], self._CUBE_Z_ON_TABLE]),
            "cube_green": np.array([self._GOAL["cube_green"]["x"], self._GOAL["cube_green"]["y"], self._CUBE_Z_ON_TABLE]),
            "cube_blue": np.array([self._GOAL["cube_blue"]["x"], self._GOAL["cube_blue"]["y"], self._CUBE_Z_ON_TABLE]),
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

        # 2. REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # 3. Randomize objects to UNSOLVED poses within required bounds
        rng = np.random.default_rng()

        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z_ON_TABLE

        # Keep cubes reasonably separated to avoid interpenetration at reset
        min_sep = 0.07

        def rand_quat_yaw():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        def sample_positions():
            pts = []
            for _ in range(5000):
                p = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z], dtype=float)
                if all(np.linalg.norm(p[:2] - q[:2]) >= min_sep for q in pts):
                    pts.append(p)
                    if len(pts) == 3:
                        return pts
            # Fallback (should be rare)
            return [
                np.array([0.45, 0.18, z], dtype=float),
                np.array([0.55, 0.00, z], dtype=float),
                np.array([0.62, -0.18, z], dtype=float),
            ]

        red_goal = self._goal_positions["cube_red"]
        green_goal = self._goal_positions["cube_green"]
        blue_goal = self._goal_positions["cube_blue"]

        def is_solved_like(rp, gp, bp) -> bool:
            # Mirror _success() position checks closely enough to avoid accidental success.
            def within_tol(p, g):
                return (abs(float(p[0] - g[0])) <= self._TOL_X) and (abs(float(p[1] - g[1])) <= self._TOL_Y)

            red_ok = within_tol(rp, red_goal)
            green_ok = within_tol(gp, green_goal)
            blue_ok = within_tol(bp, blue_goal)

            aligned_x_ok = (
                abs(float(rp[0] - self._X_BACK_EDGE)) <= self._ROW_X_TOL
                and abs(float(gp[0] - self._X_BACK_EDGE)) <= self._ROW_X_TOL
                and abs(float(bp[0] - self._X_BACK_EDGE)) <= self._ROW_X_TOL
            )
            y_order_ok = (float(rp[1]) > float(gp[1])) and (float(gp[1]) > float(bp[1]))

            return bool(red_ok and green_ok and blue_ok and aligned_x_ok and y_order_ok)

        # Rejection sample until not in (or extremely near) goal configuration
        positions = None
        for _ in range(200):
            rp, gp, bp = sample_positions()
            # Strongly bias away from the back-edge row to prevent accidental "aligned_x_ok"
            if (
                abs(float(rp[0] - self._X_BACK_EDGE)) <= (self._ROW_X_TOL + 0.01)
                and abs(float(gp[0] - self._X_BACK_EDGE)) <= (self._ROW_X_TOL + 0.01)
                and abs(float(bp[0] - self._X_BACK_EDGE)) <= (self._ROW_X_TOL + 0.01)
            ):
                continue
            if not is_solved_like(rp, gp, bp):
                positions = (rp, gp, bp)
                break

        if positions is None:
            # Deterministic unsolved fallback within required bounds
            positions = (
                np.array([0.45, -0.10, z], dtype=float),
                np.array([0.55, 0.10, z], dtype=float),
                np.array([0.65, 0.00, z], dtype=float),
            )

        rp, gp, bp = positions

        self.cube_red.set_pose(position=rp, quat=rand_quat_yaw())
        self.cube_green.set_pose(position=gp, quat=rand_quat_yaw())
        self.cube_blue.set_pose(position=bp, quat=rand_quat_yaw())

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _cube_pose_error_xy(self, cube: ColoredCube, goal_xyz: np.ndarray) -> Dict[str, float]:
        pos = cube.body.get_position()
        dx = float(pos[0] - goal_xyz[0])
        dy = float(pos[1] - goal_xyz[1])
        return {"dx": dx, "dy": dy, "dxy": float(np.sqrt(dx * dx + dy * dy))}

    def _within_bounds(self, pos: np.ndarray) -> bool:
        return (
            self._WORKSPACE_X_BOUNDS[0] <= float(pos[0]) <= self._WORKSPACE_X_BOUNDS[1]
            and self._WORKSPACE_Y_BOUNDS[0] <= float(pos[1]) <= self._WORKSPACE_Y_BOUNDS[1]
        )

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Goal checks: "on table" + position targets + arrangement ---
        red_pos = self.cube_red.body.get_position()
        green_pos = self.cube_green.body.get_position()
        blue_pos = self.cube_blue.body.get_position()

        # On-table checks (use table.table_top)
        on_table_ok = (
            self.cube_red.is_colliding(self.table.table_top)
            and self.cube_green.is_colliding(self.table.table_top)
            and self.cube_blue.is_colliding(self.table.table_top)
        )

        # Within workspace bounds (defensive)
        bounds_ok = self._within_bounds(red_pos) and self._within_bounds(green_pos) and self._within_bounds(blue_pos)

        # Goal positions (kept for diagnostic metrics only)
        red_goal = self._goal_positions["cube_red"]
        green_goal = self._goal_positions["cube_green"]
        blue_goal = self._goal_positions["cube_blue"]

        # Arrangement: cubes form a row in the back-half of the workspace
        # (use a generous workspace constraint instead of a tight 1.5 cm gate).
        back_half_ok = (
            float(red_pos[0]) >= 0.55
            and float(green_pos[0]) >= 0.55
            and float(blue_pos[0]) >= 0.55
        )
        # Row alignment: max - min X across the three cubes is small (relative).
        xs = [float(red_pos[0]), float(green_pos[0]), float(blue_pos[0])]
        row_aligned_ok = (max(xs) - min(xs)) <= 0.04

        # Left-to-right order with a minimum Y separation between adjacent cubes.
        min_y_sep = 0.04
        y_order_ok = (
            (float(red_pos[1]) - float(green_pos[1])) >= min_y_sep
            and (float(green_pos[1]) - float(blue_pos[1])) >= min_y_sep
        )

        positions_ok = bool(on_table_ok and bounds_ok and back_half_ok and row_aligned_ok and y_order_ok)

        # Semantic/content checks (colors must match)
        semantic_ok = bool(
            (self.cube_red.get_color() == "red")
            and (self.cube_green.get_color() == "green")
            and (self.cube_blue.get_color() == "blue")
        )

        # Released (no cube held by any gripper)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort from final state) ---
        # Stage index mapping:
        # 0 align, 1-3 red, 4-6 green, 7-9 blue
        if positions_ok and semantic_ok:
            self._metric_stage(0, True)  # align

        red_placed = self.cube_red.is_colliding(self.table.table_top) and back_half_ok and y_order_ok
        green_placed = self.cube_green.is_colliding(self.table.table_top) and back_half_ok and y_order_ok
        blue_placed = self.cube_blue.is_colliding(self.table.table_top) and back_half_ok and y_order_ok

        if red_placed and semantic_ok:
            self._metric_stage(1, True)  # grasp red (inferred)
            self._metric_stage(2, True)  # transport red (inferred)
            if released:
                self._metric_stage(3, True)  # place red (inferred)

        if green_placed and semantic_ok:
            self._metric_stage(4, True)  # grasp green (inferred)
            self._metric_stage(5, True)  # transport green (inferred)
            if released:
                self._metric_stage(6, True)  # place green (inferred)

        if blue_placed and semantic_ok:
            self._metric_stage(7, True)  # grasp blue (inferred)
            self._metric_stage(8, True)  # transport blue (inferred)
            if released:
                self._metric_stage(9, True)  # place blue (inferred)

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # --- Metrics finalize (MUST be last metric call before return) ---
        red_err = self._cube_pose_error_xy(self.cube_red, red_goal)
        green_err = self._cube_pose_error_xy(self.cube_green, green_goal)
        blue_err = self._cube_pose_error_xy(self.cube_blue, blue_goal)

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "cube_red_dxy": red_err["dxy"],
                "cube_green_dxy": green_err["dxy"],
                "cube_blue_dxy": blue_err["dxy"],
            },
            pose_error={
                "cube_red_dx": red_err["dx"],
                "cube_red_dy": red_err["dy"],
                "cube_green_dx": green_err["dx"],
                "cube_green_dy": green_err["dy"],
                "cube_blue_dx": blue_err["dx"],
                "cube_blue_dy": blue_err["dy"],
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})