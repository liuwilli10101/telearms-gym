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


class CubeSurroundTargetTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes around a green goal patch so that one cube is placed
    in front of the patch, one behind it, one to the left, and one to the right.

    Sequential Sub-Goals:
        1. Grasp the red cube with either gripper.
        2. Move the red cube toward the front-of-patch placement region.
        3. Place and release the red cube in front of the green goal patch.
        4. Grasp the blue cube with either gripper.
        5. Move the blue cube toward the behind-the-patch placement region.
        6. Place and release the blue cube behind the green goal patch.
        7. Grasp the yellow cube with either gripper.
        8. Move the yellow cube toward the left-of-patch placement region.
        9. Place and release the yellow cube to the left of the green goal patch.
        10. Grasp the orange cube with either gripper.
        11. Move the orange cube toward the right-of-patch placement region.
        12. Place and release the orange cube to the right of the green goal patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants ---
    _NUM_SUBGOALS = 12
    _CUBE_Z = 0.97
    _PATCH_Z = 0.951

    # Goal specification (meters)
    _FRONT_BEHIND = dict(min_d=0.06, max_d=0.12, lateral_tol=0.04)
    _LEFT_RIGHT = dict(min_d=0.06, max_d=0.12, longitudinal_tol=0.04)

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create required assets ONLY
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.yellow_cube = ColoredCube(self._mojo)
        self.orange_cube = ColoredCube(self._mojo)

        # Maintain a consistent cube list for metric tracking
        self.cubes = [self.red_cube, self.blue_cube, self.yellow_cube, self.orange_cube]

        # ⚠️ REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.yellow_cube.set_color("yellow")
        self.orange_cube.set_color("orange")

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

        # 2. Deterministic RNG (kept for pipeline compatibility; do not use global RNG)
        rng = np.random.default_rng(self.seed)

        # --- Workspace bounds ---
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # 3. Randomize patch pose (keep margins so nominal goal offsets remain in-bounds)
        margin = 0.13  # > max goal offset (0.12)
        patch_x = float(rng.uniform(x_min + margin, x_max - margin))
        patch_y = float(rng.uniform(y_min + margin, y_max - margin))
        patch_xy = np.array([patch_x, patch_y], dtype=float)
        patch_pos = np.array([patch_xy[0], patch_xy[1], self._PATCH_Z], dtype=float)
        self.green_goal_patch.body.set_position(patch_pos)
        self.green_goal_patch.body.set_quaternion(np.array([0.0, 0.0, 0.0, 1.0], dtype=float))

        # Nominal goal poses (used by _success() for metric pose error; keep consistent)
        offset = 0.09
        self._goal_positions = {
            "green_goal_patch": patch_pos,
            "red_cube": np.array([patch_xy[0] + offset, patch_xy[1], self._CUBE_Z], dtype=float),     # in front (+X)
            "blue_cube": np.array([patch_xy[0] - offset, patch_xy[1], self._CUBE_Z], dtype=float),    # behind (-X)
            "yellow_cube": np.array([patch_xy[0], patch_xy[1] + offset, self._CUBE_Z], dtype=float),  # left (+Y)
            "orange_cube": np.array([patch_xy[0], patch_xy[1] - offset, self._CUBE_Z], dtype=float),  # right (-Y)
        }

        # ⚠️ REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")
        self.yellow_cube.set_color("yellow")
        self.orange_cube.set_color("orange")

        # 4. Place cubes in RANDOM, UNSOLVED positions (guarantee _success() == False after reset)
        min_sep = 0.065  # keep cubes reasonably separated

        def _rand_yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        placed_xy: list[np.ndarray] = []

        def _far_enough(xy: np.ndarray) -> bool:
            for p in placed_xy:
                if float(np.linalg.norm(xy - p)) < min_sep:
                    return False
            return True

        def _sample_xy_avoiding(success_region_fn, max_tries: int = 2000) -> np.ndarray:
            # sample until NOT in that cube's goal region AND separated from other cubes
            for _ in range(max_tries):
                xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

                # keep away from patch center a bit (avoid accidental goal satisfactions)
                if float(np.linalg.norm(xy - patch_xy)) < 0.05:
                    continue

                pos3 = np.array([xy[0], xy[1], self._CUBE_Z], dtype=float)
                if success_region_fn(pos3, patch_pos):
                    continue
                if not _far_enough(xy):
                    continue
                return xy
            # Fallback (still keep separation best-effort): place near a corner
            return np.array([x_min + 0.02, y_max - 0.02], dtype=float)

        # For each cube, explicitly avoid its own success region
        red_xy = _sample_xy_avoiding(self._check_in_front_of)
        placed_xy.append(red_xy)
        blue_xy = _sample_xy_avoiding(self._check_behind)
        placed_xy.append(blue_xy)
        yellow_xy = _sample_xy_avoiding(self._check_left_of)
        placed_xy.append(yellow_xy)
        orange_xy = _sample_xy_avoiding(self._check_right_of)
        placed_xy.append(orange_xy)

        self.red_cube.set_pose(
            position=np.array([red_xy[0], red_xy[1], self._CUBE_Z], dtype=float),
            quat=_rand_yaw_quat(),
        )
        self._mojo.physics.forward()

        self.blue_cube.set_pose(
            position=np.array([blue_xy[0], blue_xy[1], self._CUBE_Z], dtype=float),
            quat=_rand_yaw_quat(),
        )
        self._mojo.physics.forward()

        self.yellow_cube.set_pose(
            position=np.array([yellow_xy[0], yellow_xy[1], self._CUBE_Z], dtype=float),
            quat=_rand_yaw_quat(),
        )
        self._mojo.physics.forward()

        self.orange_cube.set_pose(
            position=np.array([orange_xy[0], orange_xy[1], self._CUBE_Z], dtype=float),
            quat=_rand_yaw_quat(),
        )
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # Keep rng referenced (avoid "unused" lint patterns; also useful for future perturbations)
        self._rng = rng

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy(pos: np.ndarray) -> np.ndarray:
        return np.array([pos[0], pos[1]], dtype=float)

    def _check_in_front_of(self, subj_pos: np.ndarray, tgt_pos: np.ndarray) -> bool:
        dx = float(subj_pos[0] - tgt_pos[0])
        dy = float(subj_pos[1] - tgt_pos[1])
        return (self._FRONT_BEHIND["min_d"] <= dx <= self._FRONT_BEHIND["max_d"]) and (
            abs(dy) <= self._FRONT_BEHIND["lateral_tol"]
        )

    def _check_behind(self, subj_pos: np.ndarray, tgt_pos: np.ndarray) -> bool:
        dx = float(tgt_pos[0] - subj_pos[0])  # positive if subj is behind target
        dy = float(subj_pos[1] - tgt_pos[1])
        return (self._FRONT_BEHIND["min_d"] <= dx <= self._FRONT_BEHIND["max_d"]) and (
            abs(dy) <= self._FRONT_BEHIND["lateral_tol"]
        )

    def _check_left_of(self, subj_pos: np.ndarray, tgt_pos: np.ndarray) -> bool:
        dy = float(subj_pos[1] - tgt_pos[1])  # positive if subj is left (+Y)
        dx = float(subj_pos[0] - tgt_pos[0])
        return (self._LEFT_RIGHT["min_d"] <= dy <= self._LEFT_RIGHT["max_d"]) and (
            abs(dx) <= self._LEFT_RIGHT["longitudinal_tol"]
        )

    def _check_right_of(self, subj_pos: np.ndarray, tgt_pos: np.ndarray) -> bool:
        dy = float(tgt_pos[1] - subj_pos[1])  # positive if subj is right (-Y)
        dx = float(subj_pos[0] - tgt_pos[0])
        return (self._LEFT_RIGHT["min_d"] <= dy <= self._LEFT_RIGHT["max_d"]) and (
            abs(dx) <= self._LEFT_RIGHT["longitudinal_tol"]
        )

    def _is_released(self, obj) -> bool:
        return not any(self.robot.is_gripper_holding_object(obj, s) for s in self.robot.grippers)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "green_goal_patch"):
            return False

        # Positions
        patch_pos = self.green_goal_patch.body.get_position()

        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()
        yellow_pos = self.yellow_cube.body.get_position()
        orange_pos = self.orange_cube.body.get_position()

        # Basic "on-table height" sanity (avoid airborne success)
        z_tol = 0.03
        red_z_ok = abs(float(red_pos[2] - self._CUBE_Z)) <= z_tol
        blue_z_ok = abs(float(blue_pos[2] - self._CUBE_Z)) <= z_tol
        yellow_z_ok = abs(float(yellow_pos[2] - self._CUBE_Z)) <= z_tol
        orange_z_ok = abs(float(orange_pos[2] - self._CUBE_Z)) <= z_tol

        # Goal relations
        red_ok = self._check_in_front_of(red_pos, patch_pos) and red_z_ok
        blue_ok = self._check_behind(blue_pos, patch_pos) and blue_z_ok
        yellow_ok = self._check_left_of(yellow_pos, patch_pos) and yellow_z_ok
        orange_ok = self._check_right_of(orange_pos, patch_pos) and orange_z_ok

        positions_ok = bool(red_ok and blue_ok and yellow_ok and orange_ok)

        # Semantic/color check (ensure the right cube is the right color)
        semantic_ok = (
            (self.red_cube.get_color() == "red")
            and (self.blue_cube.get_color() == "blue")
            and (self.yellow_cube.get_color() == "yellow")
            and (self.orange_cube.get_color() == "orange")
        )

        # Released: no cube held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # --- Subgoal metric staging (approximate via holding/placement state) ---
        # red: (0 grasp, 1 transport, 2 place)
        red_holding = not self._is_released(self.red_cube)
        if red_holding:
            self._metric_stage(0, True)
        if self._check_in_front_of(red_pos, patch_pos):
            self._metric_stage(1, True)
        if red_ok and self._is_released(self.red_cube):
            self._metric_stage(2, True)

        # blue: (3,4,5)
        blue_holding = not self._is_released(self.blue_cube)
        if blue_holding:
            self._metric_stage(3, True)
        if self._check_behind(blue_pos, patch_pos):
            self._metric_stage(4, True)
        if blue_ok and self._is_released(self.blue_cube):
            self._metric_stage(5, True)

        # yellow: (6,7,8)
        yellow_holding = not self._is_released(self.yellow_cube)
        if yellow_holding:
            self._metric_stage(6, True)
        if self._check_left_of(yellow_pos, patch_pos):
            self._metric_stage(7, True)
        if yellow_ok and self._is_released(self.yellow_cube):
            self._metric_stage(8, True)

        # orange: (9,10,11)
        orange_holding = not self._is_released(self.orange_cube)
        if orange_holding:
            self._metric_stage(9, True)
        if self._check_right_of(orange_pos, patch_pos):
            self._metric_stage(10, True)
        if orange_ok and self._is_released(self.orange_cube):
            self._metric_stage(11, True)

        # Final success condition
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: target distances and pose error to nominal goal positions
        def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.linalg.norm(self._xy(a) - self._xy(b)))

        def _pose_err(name: str, actual: np.ndarray) -> float:
            goal = self._goal_positions.get(name, actual)
            return float(np.linalg.norm(actual - goal))

        target_distance = {
            "red_to_patch_xy": _xy_dist(red_pos, patch_pos),
            "blue_to_patch_xy": _xy_dist(blue_pos, patch_pos),
            "yellow_to_patch_xy": _xy_dist(yellow_pos, patch_pos),
            "orange_to_patch_xy": _xy_dist(orange_pos, patch_pos),
        }
        pose_error = {
            "red_pose_err": _pose_err("red_cube", red_pos),
            "blue_pose_err": _pose_err("blue_cube", blue_pos),
            "yellow_pose_err": _pose_err("yellow_cube", yellow_pos),
            "orange_pose_err": _pose_err("orange_cube", orange_pos),
        }

        # Finalize metrics at END (must be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})