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

class CubeLeftShiftTask(RoboEvalEnv, MetricRolloutEval):
    """
    cube-left-shift: Translate a single colored cube 15 cm to the left (+Y)
    from its initial position while keeping it within workspace bounds.

    Sequential Sub-Goals:
        1. Acquire a stable grasp on the cube (either hand).
        2. Lift the cube slightly to clear the table for unobstructed translation.
        3. Move the cube 15 cm in the +Y direction (left) while keeping x approximately constant.
        4. Place the cube down at the new left-shifted location.
        5. Release the cube without disturbing its final placement.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal spec constants
    _DELTA = np.array([0.0, 0.15, 0.0], dtype=float)  # +Y is left
    _TOL = np.array([0.02, 0.02, 0.03], dtype=float)
    _BOUNDS_X = (0.40, 0.70)
    _BOUNDS_Y = (-0.25, 0.25)
    _TABLE_CUBE_Z = 0.97

    def _initialize_env(self):
        # Props from preset
        self.table = self._preset.get_props(Table)[0]

        # Create assets (ONLY ColoredCube is allowed)
        self.colored_cube_0 = ColoredCube(self._mojo)
        self.colored_cube_0.set_color("blue")  # REQUIRED in _initialize_env

        # Convenience list for metrics
        self.cubes = [self.colored_cube_0]

        # Internal state (filled on reset)
        self._start_pos = None
        self._goal_pos = None

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

        # 2. Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # REQUIRED in _on_reset
        self.colored_cube_0.set_color("blue")

        # 3. Randomize an UNSOLVED initial pose while keeping the goal definition valid/in-bounds.
        #    Store _start_pos as the task's reference "start", and define _goal_pos = _start_pos + DELTA.
        #    Place the cube at a different pose so disp != DELTA (thus _success() is False after reset).
        start_x = float(rng.uniform(self._BOUNDS_X[0] + 0.03, self._BOUNDS_X[1] - 0.03))
        start_y = float(rng.uniform(self._BOUNDS_Y[0] + 0.03, self._BOUNDS_Y[1] - self._DELTA[1] - 0.03))
        start_y = float(np.clip(start_y, self._BOUNDS_Y[0], self._BOUNDS_Y[1] - self._DELTA[1]))

        self._start_pos = np.array([start_x, start_y, self._TABLE_CUBE_Z], dtype=float)
        self._goal_pos = self._start_pos + np.array([0.0, self._DELTA[1], 0.0], dtype=float)

        # Place cube at an UNSOLVED pose (random, in-bounds, on table) that is not within tolerance of the goal.
        # Sample until we're confidently outside the success tolerance region.
        max_tries = 50
        placed = False
        for _ in range(max_tries):
            pos_x = float(rng.uniform(self._BOUNDS_X[0] + 0.03, self._BOUNDS_X[1] - 0.03))
            pos_y = float(rng.uniform(self._BOUNDS_Y[0] + 0.03, self._BOUNDS_Y[1] - 0.03))
            pos = np.array([pos_x, pos_y, self._TABLE_CUBE_Z], dtype=float)

            disp = pos - self._start_pos
            disp_ok = np.all(np.abs(disp - self._DELTA) <= self._TOL)

            if not disp_ok:
                yaw = float(rng.uniform(-np.pi, np.pi))
                quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
                self.colored_cube_0.set_pose(position=pos, quat=quat)
                placed = True
                break

        if not placed:
            # Fallback: ensure unsolved by placing at the start pose (disp = 0 != DELTA).
            yaw = float(rng.uniform(-np.pi, np.pi))
            quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)
            self.colored_cube_0.set_pose(position=self._start_pos.copy(), quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False (5 subgoals)
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if self._start_pos is None or self._goal_pos is None:
            return False

        cube = self.colored_cube_0
        pos = np.array(cube.body.get_position(), dtype=float)

        # Bounds (task-specific)
        in_bounds = (
            (self._BOUNDS_X[0] <= pos[0] <= self._BOUNDS_X[1]) and
            (self._BOUNDS_Y[0] <= pos[1] <= self._BOUNDS_Y[1])
        )

        # Relative translation check (compare to start pose stored at reset)
        disp = pos - self._start_pos
        disp_ok = np.all(np.abs(disp - self._DELTA) <= self._TOL)

        # Also ensure final pose is on/at the table height (task says "place down")
        on_table = cube.is_colliding(self.table.table_top)
        z_ok = abs(pos[2] - self._TABLE_CUBE_Z) <= self._TOL[2]

        # Release check (generic gripper iteration; no HandSide)
        holding = any(self.robot.is_gripper_holding_object(cube, g) for g in self.robot.grippers)
        released = not holding

        positions_ok = bool(in_bounds and disp_ok and z_ok and on_table)
        semantic_ok = True  # no semantic constraints for this task

        # Subgoal staging (best-effort signals)
        # 1) grasp
        if holding:
            self._metric_stage(0, True)

        # 2) lift
        if holding and (pos[2] >= self._TABLE_CUBE_Z + 0.02) and (not on_table):
            self._metric_stage(1, True)

        # 3) transport (while held, close to target displacement)
        if holding and (abs(disp[1] - self._DELTA[1]) <= self._TOL[1]) and (abs(disp[0] - self._DELTA[0]) <= self._TOL[0]):
            self._metric_stage(2, True)

        # 4) place (on table and close to target)
        if positions_ok:
            self._metric_stage(3, True)

        # 5) release
        if positions_ok and released:
            self._metric_stage(4, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics finalization
        pos_err = float(np.linalg.norm(pos - self._goal_pos))
        disp_err = float(np.linalg.norm(disp - self._DELTA))
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "colored_cube_0_pos_to_goal_pos": pos_err,
                "colored_cube_0_disp_to_target_disp": disp_err,
            },
            pose_error={
                "colored_cube_0_x_err": float(pos[0] - self._goal_pos[0]),
                "colored_cube_0_y_err": float(pos[1] - self._goal_pos[1]),
                "colored_cube_0_z_err": float(pos[2] - self._goal_pos[2]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})