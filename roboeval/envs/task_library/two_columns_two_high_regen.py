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

class SideBySideColumnStacking(RoboEvalEnv, MetricRolloutEval):
    """
    Build two vertical columns using four colored cubes, each column two cubes high.
    The two columns must be placed side by side along the X (front/back) axis and
    aligned in Y (left/right).

    Sequential Sub-Goals:
        1. Grasp a base cube for the first column. (tracked spatially)
        2. Place the first base cube stably on the table.
        3. Grasp the cube that will become the top of the first column. (tracked spatially)
        4. Place the top cube onto the first base cube to complete column 1.
        5. Grasp a base cube for the second column. (tracked spatially)
        6. Place the second base cube on the table with similar Y to the first base cube but separated in X.
        7. Grasp the cube that will become the top of the second column. (tracked spatially)
        8. Place the top cube onto the second base cube to complete column 2.
        9. Adjust placement so the two column bases are aligned in Y and separated along X.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # ---- Task constants (per spec) ----
    _CUBE_SIZE = 0.04
    _BASE_Z = 0.97
    _STACK_DZ = 0.04

    _Y_ALIGN_TOL = 0.03
    _X_MIN_SEP = 0.08
    _X_MAX_SEP = 0.18

    _STACK_XY_TOL = 0.02
    _STACK_DZ_TOL = 0.02

    _NUM_SUBGOALS = 9

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create required assets (ONLY ColoredCube)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)

        # REQUIRED: set_color in BOTH _initialize_env AND _on_reset
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_green.set_color("green")
        self.cube_yellow.set_color("yellow")

        self.cubes = [self.cube_red, self.cube_blue, self.cube_green, self.cube_yellow]

        # Goal layout (X is front/back, Y is left/right)
        # Two base cubes aligned in Y, separated along X.
        self._goal_blue_base = np.array([0.55, 0.00, self._BASE_Z])
        self._goal_yellow_base = np.array([0.67, 0.00, self._BASE_Z])  # sep=0.12 within [0.08,0.18]
        self._goal_red_top = self._goal_blue_base + np.array([0.0, 0.0, self._STACK_DZ])
        self._goal_green_top = self._goal_yellow_base + np.array([0.0, 0.0, self._STACK_DZ])

        # Initialize metrics (FULL signature REQUIRED)
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
        # REQUIRED: set_color in BOTH _initialize_env AND _on_reset
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_green.set_color("green")
        self.cube_yellow.set_color("yellow")

        # 1) Re-initialize metrics (REQUIRED on every reset)
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # 2) Deterministic RNG
        rng = np.random.default_rng(self.seed)

        # 3) Place objects in RANDOM, UNSOLVED positions within workspace bounds
        # Workspace: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._BASE_Z

        # Keep cubes separated so they don't start stacked or intersecting
        min_xy_sep = 1.5 * self._CUBE_SIZE  # conservative

        def sample_pose(existing_xy: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            for _ in range(5000):
                x = float(rng.uniform(x_lo, x_hi))
                y = float(rng.uniform(y_lo, y_hi))
                xy = np.array([x, y], dtype=float)
                if all(np.linalg.norm(xy - e) >= min_xy_sep for e in existing_xy):
                    yaw = float(rng.uniform(-np.pi, np.pi))
                    quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
                    pos = np.array([x, y, z], dtype=float)
                    return pos, quat
            # Fallback (should be rare)
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
            pos = np.array([float(rng.uniform(x_lo, x_hi)), float(rng.uniform(y_lo, y_hi)), z], dtype=float)
            return pos, quat

        existing = []
        p_blue, q_blue = sample_pose(existing); existing.append(p_blue[:2].copy())
        p_yellow, q_yellow = sample_pose(existing); existing.append(p_yellow[:2].copy())
        p_red, q_red = sample_pose(existing); existing.append(p_red[:2].copy())
        p_green, q_green = sample_pose(existing); existing.append(p_green[:2].copy())

        # Ensure NOT in goal configuration by avoiding exact goal poses (and not stacked by construction)
        eps = 1e-6
        if np.linalg.norm(p_blue - self._goal_blue_base) < eps:
            p_blue[0] = float(np.clip(p_blue[0] + 0.02, x_lo, x_hi))
        if np.linalg.norm(p_yellow - self._goal_yellow_base) < eps:
            p_yellow[0] = float(np.clip(p_yellow[0] - 0.02, x_lo, x_hi))

        # Set all cubes on the table (no initial stacking), then forward physics
        self.cube_blue.set_pose(position=p_blue, quat=q_blue)
        self.cube_yellow.set_pose(position=p_yellow, quat=q_yellow)
        self.cube_red.set_pose(position=p_red, quat=q_red)
        self.cube_green.set_pose(position=p_green, quat=q_green)
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _xy_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _is_stacked_on(self, top: ColoredCube, bottom: ColoredCube) -> bool:
        # Must be physically in contact AND vertically offset by ~one cube AND aligned in XY.
        if not top.is_colliding(bottom):
            return False

        p_top = top.body.get_position()
        p_bot = bottom.body.get_position()

        dz = float(p_top[2] - p_bot[2])
        if abs(dz - self._STACK_DZ) > self._STACK_DZ_TOL:
            return False

        if self._xy_dist(p_top, p_bot) > self._STACK_XY_TOL:
            return False

        # Top cube should not be on the table surface if stacked
        if top.is_colliding(self.table.table_top):
            return False

        return True

    def _is_on_table(self, cube: ColoredCube) -> bool:
        return bool(cube.is_colliding(self.table.table_top))

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Arrangement checks (two 2-high columns)
        col1_ok = self._is_stacked_on(self.cube_red, self.cube_blue) and self._is_on_table(self.cube_blue)
        col2_ok = self._is_stacked_on(self.cube_green, self.cube_yellow) and self._is_on_table(self.cube_yellow)

        # Relative checks for bases: aligned in Y (index 1), separated along X (index 0)
        p_blue = self.cube_blue.body.get_position()
        p_yellow = self.cube_yellow.body.get_position()

        y_aligned = abs(float(p_blue[1] - p_yellow[1])) <= self._Y_ALIGN_TOL
        x_sep = abs(float(p_blue[0] - p_yellow[0]))
        x_separated = (x_sep >= self._X_MIN_SEP) and (x_sep <= self._X_MAX_SEP)

        positions_ok = bool(col1_ok and col2_ok and y_aligned and x_separated)
        semantic_ok = True  # ColoredCube task; semantics satisfied by fixed cube identities/colors

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Subgoal staging (spatial proxies; grasp stages are approximated by placement conditions)
        # 0-1: first base cube placed
        if self._is_on_table(self.cube_blue):
            self._metric_stage(0, True)
            self._metric_stage(1, True)
        # 2-3: first column completed
        if self._is_stacked_on(self.cube_red, self.cube_blue):
            self._metric_stage(2, True)
            self._metric_stage(3, True)
        # 4-5: second base cube placed with correct relative X/Y (base placement intent)
        if self._is_on_table(self.cube_yellow):
            self._metric_stage(4, True)
            if y_aligned and x_separated:
                self._metric_stage(5, True)
        # 6-7: second column completed
        if self._is_stacked_on(self.cube_green, self.cube_yellow):
            self._metric_stage(6, True)
            self._metric_stage(7, True)
        # 8: overall alignment/separation satisfied (columns side-by-side along X, aligned in Y)
        if y_aligned and x_separated:
            self._metric_stage(8, True)

        # Final success flag
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (MUST be last metric call before return)
        target_distance = {
            "red_to_blue_top": float(
                np.linalg.norm(self.cube_red.body.get_position() - self._goal_red_top)
            ),
            "green_to_yellow_top": float(
                np.linalg.norm(self.cube_green.body.get_position() - self._goal_green_top)
            ),
            "blue_base_to_goal": float(
                np.linalg.norm(self.cube_blue.body.get_position() - self._goal_blue_base)
            ),
            "yellow_base_to_goal": float(
                np.linalg.norm(self.cube_yellow.body.get_position() - self._goal_yellow_base)
            ),
        }
        pose_error = {
            "y_alignment_error": abs(float(p_blue[1] - p_yellow[1])),
            "x_separation_error_to_midrange": abs(x_sep - 0.5 * (self._X_MIN_SEP + self._X_MAX_SEP)),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})