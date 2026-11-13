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

class TriangleCubePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Place three colored cubes on the table so they form an equilateral triangle
    around the workspace center, with each cube 8 cm from the center.

    Sequential Sub-Goals:
        1. Grasp the red cube with either gripper.
        2. Move the red cube toward its triangle vertex around the workspace center.
        3. Place the red cube down at its target vertex location.
        4. Release the red cube after placing.
        5. Grasp the green cube with either gripper.
        6. Move the green cube toward its triangle vertex around the workspace center.
        7. Place the green cube down at its target vertex location.
        8. Release the green cube after placing.
        9. Grasp the blue cube with either gripper.
        10. Move the blue cube toward its triangle vertex around the workspace center.
        11. Place the blue cube down at its target vertex location.
        12. Release the blue cube after placing.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task geometry / tolerances
    _CENTER_XY = np.array([0.55, 0.0])
    _RADIUS_M = 0.08
    _ANGLES_DEG = {"red": 0.0, "green": 120.0, "blue": 240.0}

    _POS_TOL_M = 0.015
    _Z_ON_TABLE_TOL_M = 0.02
    _PAIR_DIST_TARGET_M = 0.1386
    _PAIR_DIST_TOL_M = 0.02

    _CUBE_Z = 0.97
    _CUBE_QUAT = np.array([0.0, 0.0, 0.0, 1.0])

    _NUM_SUBGOALS = 12

    def _polar_to_xy(self, radius_m: float, angle_deg: float) -> np.ndarray:
        th = np.deg2rad(angle_deg)
        return self._CENTER_XY + np.array([radius_m * np.cos(th), radius_m * np.sin(th)])

    def _goal_xy_for_color(self, color: str) -> np.ndarray:
        return self._polar_to_xy(self._RADIUS_M, self._ANGLES_DEG[color])

    def _is_on_table(self, cube: ColoredCube) -> bool:
        # Collision with table top is the most reliable "on table" indicator.
        if not cube.is_colliding(self.table.table_top):
            return False
        z = float(cube.body.get_position()[2])
        return abs(z - self._CUBE_Z) <= self._Z_ON_TABLE_TOL_M

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset: ColoredCube)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]

        # Precompute goal positions (triangle vertices around workspace center)
        self._goal_positions = {
            "red": np.array([*self._goal_xy_for_color("red"), self._CUBE_Z]),
            "green": np.array([*self._goal_xy_for_color("green"), self._CUBE_Z]),
            "blue": np.array([*self._goal_xy_for_color("blue"), self._CUBE_Z]),
        }

        # Initialize metrics (FULL signature required)
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

        # 2. Deterministic RNG (required pattern; validation pipeline may apply randomization later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # 3. Randomize cube poses to be UNSOLVED (ensure _success() is False after reset)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z

        goals_xy = {
            "red": self._goal_positions["red"][:2].copy(),
            "green": self._goal_positions["green"][:2].copy(),
            "blue": self._goal_positions["blue"][:2].copy(),
        }

        # Keep cubes separated and away from their own goal vertices so per-cube goal checks fail.
        min_sep = 0.05
        min_goal_clearance = self._POS_TOL_M + 0.03

        def sample_xy_away_from_goal(goal_xy: np.ndarray) -> np.ndarray:
            for _ in range(5000):
                xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)
                if float(np.linalg.norm(xy - goal_xy)) >= min_goal_clearance:
                    return xy
            # Fallback: force a point far from the goal by clipping an offset
            xy = goal_xy + np.array([0.10, 0.10])
            xy[0] = float(np.clip(xy[0], x_min, x_max))
            xy[1] = float(np.clip(xy[1], y_min, y_max))
            return xy

        def far_enough(xy: np.ndarray, placed: list[np.ndarray]) -> bool:
            return all(float(np.linalg.norm(xy - p)) >= min_sep for p in placed)

        placed: list[np.ndarray] = []

        red_xy = sample_xy_away_from_goal(goals_xy["red"])
        placed.append(red_xy)

        for _ in range(5000):
            green_xy = sample_xy_away_from_goal(goals_xy["green"])
            if far_enough(green_xy, placed):
                placed.append(green_xy)
                break
        else:
            green_xy = np.array([x_min, y_min], dtype=float)
            placed.append(green_xy)

        for _ in range(5000):
            blue_xy = sample_xy_away_from_goal(goals_xy["blue"])
            if far_enough(blue_xy, placed):
                placed.append(blue_xy)
                break
        else:
            blue_xy = np.array([x_max, y_max], dtype=float)
            placed.append(blue_xy)

        # Additionally, ensure the three cubes do NOT accidentally form the target equilateral triangle.
        def tri_accidentally_solved(rxy: np.ndarray, gxy: np.ndarray, bxy: np.ndarray) -> bool:
            d_rg = float(np.linalg.norm(rxy - gxy))
            d_gb = float(np.linalg.norm(gxy - bxy))
            d_br = float(np.linalg.norm(bxy - rxy))
            return (
                abs(d_rg - self._PAIR_DIST_TARGET_M) <= self._PAIR_DIST_TOL_M
                and abs(d_gb - self._PAIR_DIST_TARGET_M) <= self._PAIR_DIST_TOL_M
                and abs(d_br - self._PAIR_DIST_TARGET_M) <= self._PAIR_DIST_TOL_M
            )

        if tri_accidentally_solved(red_xy, green_xy, blue_xy):
            # Nudge blue away deterministically within bounds.
            blue_xy = blue_xy + np.array([0.07, -0.06])
            blue_xy[0] = float(np.clip(blue_xy[0], x_min, x_max))
            blue_xy[1] = float(np.clip(blue_xy[1], y_min, y_max))

        self.cube_red.set_pose(position=np.array([red_xy[0], red_xy[1], z]), quat=self._CUBE_QUAT)
        self.cube_green.set_pose(position=np.array([green_xy[0], green_xy[1], z]), quat=self._CUBE_QUAT)
        self.cube_blue.set_pose(position=np.array([blue_xy[0], blue_xy[1], z]), quat=self._CUBE_QUAT)

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
        # CRITICAL: reset success flag at start
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_positions"):
            return False

        # --- Positional checks: cubes on table (no per-cube absolute world-XY gate) ---
        red_p = self.cube_red.body.get_position()
        green_p = self.cube_green.body.get_position()
        blue_p = self.cube_blue.body.get_position()

        on_table_ok = (
            self._is_on_table(self.cube_red)
            and self._is_on_table(self.cube_green)
            and self._is_on_table(self.cube_blue)
        )
        pos_ok = bool(on_table_ok)

        # --- Arrangement check: pairwise distances consistent with an equilateral triangle ---
        # Target side length for r=0.08: r*sqrt(3) ≈ 0.1386 (provided by spec)
        d_rg = float(np.linalg.norm(red_p[:2] - green_p[:2]))
        d_gb = float(np.linalg.norm(green_p[:2] - blue_p[:2]))
        d_br = float(np.linalg.norm(blue_p[:2] - red_p[:2]))

        tri_ok = (
            abs(d_rg - self._PAIR_DIST_TARGET_M) <= self._PAIR_DIST_TOL_M
            and abs(d_gb - self._PAIR_DIST_TARGET_M) <= self._PAIR_DIST_TOL_M
            and abs(d_br - self._PAIR_DIST_TARGET_M) <= self._PAIR_DIST_TOL_M
        )

        positions_ok = bool(pos_ok and tri_ok)

        # Optional semantic consistency (cube identities should match their intended colors)
        semantic_ok = (
            self.cube_red.get_color() == "red"
            and self.cube_green.get_color() == "green"
            and self.cube_blue.get_color() == "blue"
        )

        # Released (no grasp-based success criteria, but ensure robot isn't holding objects)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Stage completion: mark all subgoals achieved once final spatial goal is met
        if positions_ok and semantic_ok:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics
        target_distance = {
            "cube_red_to_goal_m": float(np.linalg.norm(red_p[:2] - self._goal_positions["red"][:2])),
            "cube_green_to_goal_m": float(np.linalg.norm(green_p[:2] - self._goal_positions["green"][:2])),
            "cube_blue_to_goal_m": float(np.linalg.norm(blue_p[:2] - self._goal_positions["blue"][:2])),
            "pair_rg_err_m": float(abs(d_rg - self._PAIR_DIST_TARGET_M)),
            "pair_gb_err_m": float(abs(d_gb - self._PAIR_DIST_TARGET_M)),
            "pair_br_err_m": float(abs(d_br - self._PAIR_DIST_TARGET_M)),
        }
        pose_error = {
            "cube_red_z_err_m": float(abs(float(red_p[2]) - self._CUBE_Z)),
            "cube_green_z_err_m": float(abs(float(green_p[2]) - self._CUBE_Z)),
            "cube_blue_z_err_m": float(abs(float(blue_p[2]) - self._CUBE_Z)),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})