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

class CubeProximityPlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Move three colored cubes so they end up within 10 cm of the green goal patch
    while ensuring none of the cubes are placed on top of the patch.

    Sequential Sub-Goals:
        1. Grasp cube_1 with either gripper.
        2. Place cube_1 within 10 cm (XY) of the green goal patch without overlapping it.
        3. Grasp cube_2 with either gripper.
        4. Place cube_2 within 10 cm (XY) of the green goal patch without overlapping it.
        5. Grasp cube_3 with either gripper.
        6. Place cube_3 within 10 cm (XY) of the green goal patch without overlapping it.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _NUM_CUBES = 3
    _NUM_SUBGOALS = 6

    _CUBE_REST_Z = 0.97
    _PATCH_REST_Z = 0.951

    _DIST_MAX_XY = 0.10  # within 10 cm in the table plane
    _PATCH_SIZE_XY = (0.10, 0.10)  # (width, depth)
    _CUBE_SIZE = 0.04

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create required assets ONLY
        self.cubes = [ColoredCube(self._mojo) for _ in range(self._NUM_CUBES)]
        self.green_goal_patch = GoalPatch(self._mojo, color="green", size=self._PATCH_SIZE_XY, name="target")

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        self._cube_colors = ["red", "blue", "yellow"]
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # Subgoal tracking (used for metric staging)
        self._subgoal_done = [False] * self._NUM_SUBGOALS

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

        # 2. Deterministic RNG (required pattern, even if goal placement is deterministic)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in BOTH _initialize_env() AND _on_reset()
        for cube, col in zip(self.cubes, self._cube_colors):
            cube.set_color(col)

        # 3. Place objects in RANDOM UNSOLVED configuration within workspace bounds
        # Workspace bounds (required): X [0.40, 0.70], Y [-0.25, 0.25]
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25

        # Randomize patch location
        patch_xy = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float
        )
        self.green_goal_patch.body.set_position(
            np.array([patch_xy[0], patch_xy[1], self._PATCH_REST_Z], dtype=float)
        )
        self._mojo.physics.forward()

        # Randomize cube locations such that at least one cube is NOT within 10cm XY of patch
        # Enforce all cubes are far enough from patch to guarantee _success() == False after reset.
        min_fail_dist = float(self._DIST_MAX_XY) + 0.05  # >= 0.15m ensures not close_enough
        min_cube_sep = float(self._CUBE_SIZE) + 0.02     # simple anti-overlap buffer

        def sample_cube_xy(existing_xys):
            for _ in range(2000):
                xy = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=float)

                # Ensure the cube is NOT within success radius of the patch (guarantee failure)
                if float(np.linalg.norm((xy - patch_xy).astype(float))) <= min_fail_dist:
                    continue

                # Basic cube-cube separation
                ok = True
                for e in existing_xys:
                    if float(np.linalg.norm((xy - e).astype(float))) < min_cube_sep:
                        ok = False
                        break
                if not ok:
                    continue
                return xy
            # Fallback (still within bounds and far from patch along X direction)
            xy = patch_xy.copy()
            xy[0] = np.clip(patch_xy[0] + (min_fail_dist + 0.02), x_min, x_max)
            xy[1] = np.clip(patch_xy[1], y_min, y_max)
            return xy

        cube_xys = []
        for _ in range(self._NUM_CUBES):
            cube_xys.append(sample_cube_xy(cube_xys))

        goal_positions = [
            np.array([xy[0], xy[1], self._CUBE_REST_Z], dtype=float) for xy in cube_xys
        ]

        for cube, pos in zip(self.cubes, goal_positions):
            cube.set_pose(position=pos, quat=np.array([0.0, 0.0, 0.0, 1.0]))
            self._mojo.physics.forward()

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        self._subgoal_done = [False] * self._NUM_SUBGOALS
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        # Store for metrics
        self._patch_xy = patch_xy.copy()
        self._goal_positions = goal_positions

        # (Optional) touch RNG to keep deterministic usage consistent across tasks
        _ = rng.random()

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()

        # Update subgoal stages (best-effort, without requiring full event history)
        if hasattr(self, "cubes") and hasattr(self, "green_goal_patch"):
            for i, cube in enumerate(self.cubes):
                grasp_stage = 2 * i
                place_stage = 2 * i + 1

                held_now = any(
                    self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers
                )
                if held_now and not self._subgoal_done[grasp_stage]:
                    self._subgoal_done[grasp_stage] = True
                    self._metric_stage(grasp_stage, True)

                placed_ok = self._cube_meets_goal_conditions(cube)
                if placed_ok and (not held_now) and self._subgoal_done[grasp_stage] and not self._subgoal_done[place_stage]:
                    self._subgoal_done[place_stage] = True
                    self._metric_stage(place_stage, True)

        self._metric_step()

    def _cube_meets_goal_conditions(self, cube: ColoredCube) -> bool:
        """Cube must be within 10cm in XY of patch center, and not overlap patch region in XY."""
        patch_pos = self.green_goal_patch.body.get_position()
        cube_pos = cube.body.get_position()

        dxy = np.linalg.norm((cube_pos[:2] - patch_pos[:2]).astype(float))
        close_enough = dxy <= float(self._DIST_MAX_XY)

        # Region overlap check (axis-aligned): overlap if both |dx| and |dy| are within summed half-extents.
        patch_half_x = float(self._PATCH_SIZE_XY[0]) / 2.0
        patch_half_y = float(self._PATCH_SIZE_XY[1]) / 2.0
        cube_half = float(self._CUBE_SIZE) / 2.0
        sum_half_x = patch_half_x + cube_half  # margin = 0.0
        sum_half_y = patch_half_y + cube_half  # margin = 0.0

        dx = abs(float(cube_pos[0] - patch_pos[0]))
        dy = abs(float(cube_pos[1] - patch_pos[1]))
        overlaps_patch_xy = (dx <= sum_half_x) and (dy <= sum_half_y)

        not_on_patch = not overlaps_patch_xy
        return bool(close_enough and not_on_patch)

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "green_goal_patch"):
            return False

        # Check goal conditions for all cubes
        positions_ok = all(self._cube_meets_goal_conditions(c) for c in self.cubes)

        # No semantic constraints beyond colors (colors are fixed but not part of success)
        semantic_ok = True

        # Must not be holding any cube at success
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        self._success_check = bool(positions_ok and semantic_ok and released)

        # If fully successful, mark all subgoals complete (final state)
        if self._success_check:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Metrics
        patch_pos = self.green_goal_patch.body.get_position()
        target_distance = {}
        for i, cube in enumerate(self.cubes):
            p = cube.body.get_position()
            dxy = float(np.linalg.norm((p[:2] - patch_pos[:2]).astype(float)))
            target_distance[f"cube_{i+1}_to_patch_xy"] = dxy

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error={},  # not used for this task
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})