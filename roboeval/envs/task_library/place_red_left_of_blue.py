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


class RedLeftOfBluePlacementTask(RoboEvalEnv, MetricRolloutEval):
    """Place red to the left of blue.

Success Criteria:
- The red cube is left of the blue cube by the configured positive-Y separation.
- Both cubes are on the table.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _MIN_SEPARATION_Y = 0.03
    _TABLE_Z_CUBE = 0.97
    _POSE_TOL = 0.014  # Provided task tolerance guideline (meters)

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Assets (MUST be ONLY ColoredCube)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

        # Seed goal poses (also set definitively in _on_reset)
        self._goal_blue_pos = np.array([0.55, -0.06, self._TABLE_Z_CUBE])
        self._goal_red_pos = np.array([0.55, +0.06, self._TABLE_Z_CUBE])
        self._goal_quat = np.array([1.0, 0.0, 0.0, 0.0])

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

        # 2) Deterministic RNG (required pattern; may be used by external randomization)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # Keep goal poses (used for metrics); randomize only initial poses.
        self._goal_blue_pos = np.array([0.55, -0.06, self._TABLE_Z_CUBE])
        self._goal_red_pos = np.array([0.55, +0.06, self._TABLE_Z_CUBE])

        # 3) Place objects in RANDOM unsolved positions within workspace bounds.
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97.
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._TABLE_Z_CUBE

        # Ensure the task is UNSOLVED after reset: enforce NOT(left_of_ok)
        # i.e., red_pos[1] - blue_pos[1] < _MIN_SEPARATION_Y
        # Also enforce a small absolute separation to avoid overlap/instability.
        min_abs_sep = 0.06  # keep cubes apart for stability
        max_tries = 200
        red_pos = None
        blue_pos = None

        for _ in range(max_tries):
            # Sample blue uniformly
            bx = float(rng.uniform(x_lo, x_hi))
            by = float(rng.uniform(y_lo, y_hi))

            # Sample red such that it is NOT left-of by required margin (unsolved)
            # Pick a negative (or small) delta so red is at/below blue in Y.
            dy = float(rng.uniform(-0.18, self._MIN_SEPARATION_Y - 0.005))  # strictly < MIN_SEPARATION_Y
            ry = by + dy

            # Clamp to bounds (resample if out of bounds)
            if not (y_lo <= ry <= y_hi):
                continue

            # Sample red x independently; ensure not too close in XY
            rx = float(rng.uniform(x_lo, x_hi))
            if np.hypot(rx - bx, ry - by) < min_abs_sep:
                continue

            blue_pos = np.array([bx, by, z], dtype=float)
            red_pos = np.array([rx, ry, z], dtype=float)

            # Explicitly ensure unsolved relation
            if (red_pos[1] - blue_pos[1]) < float(self._MIN_SEPARATION_Y):
                break
        else:
            # Fallback deterministic unsolved placement within bounds
            blue_pos = np.array([0.55, 0.10, z], dtype=float)
            red_pos = np.array([0.55, 0.00, z], dtype=float)

        # Random yaw rotations (stable)
        def yaw_quat(yaw: float) -> np.ndarray:
            return Quaternion(axis=[0.0, 0.0, 1.0], radians=float(yaw)).elements.astype(float)

        blue_quat = yaw_quat(rng.uniform(-np.pi, np.pi))
        red_quat = yaw_quat(rng.uniform(-np.pi, np.pi))

        # Place one-by-one for stability
        self.blue_cube.set_pose(position=blue_pos.copy(), quat=blue_quat.copy())
        self._mojo.physics.forward()
        self.red_cube.set_pose(position=red_pos.copy(), quat=red_quat.copy())
        self._mojo.physics.forward()

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()

        # --- Goal condition: red is left of blue along table Y with minimum separation ---
        # Left in this environment/camera convention corresponds to +Y.
        left_of_ok = (red_pos[1] - blue_pos[1]) >= float(self._MIN_SEPARATION_Y)

        # --- Both cubes resting stably on the table surface ---
        red_on_table = (
            self.red_cube.is_colliding(self.table.table_top)
            and abs(float(red_pos[2]) - self._TABLE_Z_CUBE) <= 0.02
        )
        blue_on_table = (
            self.blue_cube.is_colliding(self.table.table_top)
            and abs(float(blue_pos[2]) - self._TABLE_Z_CUBE) <= 0.02
        )

        positions_ok = bool(left_of_ok and red_on_table and blue_on_table)

        # No semantic constraints beyond color (already fixed by construction)
        semantic_ok = True

        # Released gripper condition
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # --- Subgoal staging (best-effort proxies for rollout metrics) ---
        # 0) Optional stabilize blue
        blue_grasped = any(
            self.robot.is_gripper_holding_object(self.blue_cube, s) for s in self.robot.grippers
        )
        if blue_grasped:
            self._metric_stage(0, True)

        # 1) Grasp red
        red_grasped = any(
            self.robot.is_gripper_holding_object(self.red_cube, s) for s in self.robot.grippers
        )
        if red_grasped:
            self._metric_stage(1, True)

        # 2) Lift red
        red_lifted = (float(red_pos[2]) >= (self._TABLE_Z_CUBE + 0.02)) and (
            not self.red_cube.is_colliding(self.table.table_top)
        )
        if red_lifted:
            self._metric_stage(2, True)

        # 3) Transport red to left side (can be satisfied while lifted)
        if left_of_ok:
            self._metric_stage(3, True)

        # 4) Place red on table in left-of relation
        if left_of_ok and red_on_table:
            self._metric_stage(4, True)

        # 5) Release after placement
        if left_of_ok and red_on_table and released:
            self._metric_stage(5, True)

        # Set success flag
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_goal": float(np.linalg.norm(red_pos - self._goal_red_pos)),
                "blue_to_goal": float(np.linalg.norm(blue_pos - self._goal_blue_pos)),
                "y_separation_error": float(
                    max(0.0, self._MIN_SEPARATION_Y - (float(red_pos[1]) - float(blue_pos[1])))
                ),
            },
            pose_error={
                "red_pos_err": float(np.linalg.norm(red_pos - self._goal_red_pos)),
                "blue_pos_err": float(np.linalg.norm(blue_pos - self._goal_blue_pos)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})