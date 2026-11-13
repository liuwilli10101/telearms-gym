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


class BlueOnRedStackToGreenTarget(RoboEvalEnv, MetricRolloutEval):
    """
    Stack the blue cube on top of the red cube, then move the two-cube stack onto
    the green goal patch.

    Sequential Sub-Goals:
        1. Grasp the blue cube with either gripper.
        2. Lift the blue cube clear of the table to enable stacking.
        3. Place the blue cube on top of the red cube to form a stable stack.
        4. Release the blue cube after it is stably stacked on the red cube.
        5. Move the stacked pair toward the green goal patch.
        6. Place the stack so that the red cube sits on the green target patch.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_Z = 0.97
    _PATCH_Z = 0.951
    _STACK_DZ = 0.04
    _XY_TOL = 0.014  # task constraint tolerance

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Props (MUST use ONLY allowed assets)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=(0.1, 0.1), name="target")

        # REQUIRED: set_color() in _initialize_env
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

        # Metrics (FULL signature required)
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
        # REQUIRED: set_color() in _on_reset
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

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

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        def _rand_xy():
            return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

        def _rand_yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        # 3) Random UNSOLVED placement (ensure not stacked and red not on patch)
        patch_xy = _rand_xy()
        patch_pos = np.array([patch_xy[0], patch_xy[1], self._PATCH_Z], dtype=float)
        self.green_goal_patch.body.set_position(patch_pos)

        # Place red cube away from patch
        for _ in range(200):
            red_xy = _rand_xy()
            if float(np.linalg.norm(red_xy - patch_xy)) > 0.12:
                break
        else:
            red_xy = np.clip(patch_xy + np.array([0.15, 0.0]), [x_min, y_min], [x_max, y_max])

        red_init_pos = np.array([red_xy[0], red_xy[1], self._CUBE_Z], dtype=float)
        self.red_cube.set_pose(position=red_init_pos, quat=_rand_yaw_quat())
        self._mojo.physics.forward()

        # Place blue cube away from red (not stacked) and away from patch
        for _ in range(300):
            blue_xy = _rand_xy()
            if float(np.linalg.norm(blue_xy - red_xy)) > 0.10 and float(np.linalg.norm(blue_xy - patch_xy)) > 0.12:
                break
        else:
            blue_xy = np.clip(red_xy + np.array([0.12, 0.12]), [x_min, y_min], [x_max, y_max])

        blue_init_pos = np.array([blue_xy[0], blue_xy[1], self._CUBE_Z], dtype=float)
        self.blue_cube.set_pose(position=blue_init_pos, quat=_rand_yaw_quat())
        self._mojo.physics.forward()

        # Store goal poses for metric reporting (desired goal, not initial placement)
        red_goal_pos = np.array([patch_pos[0], patch_pos[1], self._CUBE_Z], dtype=float)
        blue_goal_pos = np.array([patch_pos[0], patch_pos[1], self._CUBE_Z + self._STACK_DZ], dtype=float)
        self._patch_goal_pos = patch_pos.copy()
        self._red_goal_pos = red_goal_pos.copy()
        self._blue_goal_pos = blue_goal_pos.copy()

        # 5) Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _xy_dist(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _is_block_on_target(self, block: ColoredCube, target: GoalPatch) -> bool:
        # Reliable GoalPatch pattern: check collision from the BLOCK side using target.collider
        if hasattr(target, "collider"):
            if not block.is_colliding(target.collider):
                return False
        else:
            if not block.is_colliding(target):
                return False

        # If a floor geom exists in some presets, avoid counting "on floor" as success.
        if hasattr(self, "_floor") and block.is_colliding(self._floor):
            return False
        return True

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()
        patch_pos = self.green_goal_patch.body.get_position()

        # Goal 1: blue cube on red cube (stacked)
        xy_close = self._xy_dist(blue_pos, red_pos) < (2.0 * self._XY_TOL)
        z_separation = float(blue_pos[2] - red_pos[2])
        z_ok = (z_separation > 0.03) and (abs(z_separation - self._STACK_DZ) < 0.03)

        stacked = (
            self.blue_cube.is_colliding(self.red_cube)
            and (not self.blue_cube.is_colliding(self.table.table_top))
            and xy_close
            and z_ok
        )

        # Goal 2: red cube (bottom of stack) on the green goal patch
        on_patch = self._is_block_on_target(self.red_cube, self.green_goal_patch)

        # Semantic/color sanity (does not add new goal types; ensures correct objects)
        color_ok = (self.red_cube.get_color() == "red") and (self.blue_cube.get_color() == "blue")

        # Released (no grasp constraint in goal spec; used only to ensure episode end condition)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in [self.red_cube, self.blue_cube]
            for s in self.robot.grippers
        )

        positions_ok = bool(stacked and on_patch)
        semantic_ok = bool(color_ok)

        # --- Subgoal staging (best-effort signals; success is purely spatial/positional) ---
        holding_blue = any(
            self.robot.is_gripper_holding_object(self.blue_cube, s) for s in self.robot.grippers
        )
        blue_lifted = (not self.blue_cube.is_colliding(self.table.table_top)) and (blue_pos[2] > self._CUBE_Z + 0.01)
        blue_released_after_stack = stacked and (not any(
            self.robot.is_gripper_holding_object(self.blue_cube, s) for s in self.robot.grippers
        ))
        moved_toward_patch = (self._xy_dist(red_pos, patch_pos) < 0.10)  # loose transport indicator
        placed_on_patch = on_patch

        self._metric_stage(0, bool(holding_blue))
        self._metric_stage(1, bool(holding_blue and blue_lifted))
        self._metric_stage(2, bool(stacked))
        self._metric_stage(3, bool(blue_released_after_stack))
        self._metric_stage(4, bool(moved_toward_patch))
        self._metric_stage(5, bool(placed_on_patch and stacked))

        # If fully solved, mark all stages solved
        if positions_ok and semantic_ok:
            for i in range(6):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalize (must be last metric call before return)
        # Provide a small set of interpretable distances/errors.
        target_distance = {
            "red_to_patch_xy": self._xy_dist(red_pos, patch_pos),
            "blue_to_red_xy": self._xy_dist(blue_pos, red_pos),
        }
        pose_error = {
            "red_z_error": float(abs(red_pos[2] - self._CUBE_Z)),
            "blue_stack_dz_error": float(abs((blue_pos[2] - red_pos[2]) - self._STACK_DZ)),
        }
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})