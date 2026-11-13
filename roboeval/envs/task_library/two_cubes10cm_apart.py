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

class CubeSeparationTask(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange two colored cubes on the tabletop so that the distance between their
    centers is exactly 10 cm. Both cubes must remain resting on the table surface.

    Sequential Sub-Goals:
        1. Grasp the first cube with either gripper.
        2. Place the first cube onto a convenient location on the tabletop.
        3. Grasp the second cube with either gripper.
        4. Adjust the second cube's placement so the cubes end up 10 cm apart on the table.
        5. Release any grasped cube(s), leaving both cubes resting on the table.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _TARGET_DISTANCE_M = 0.10
    _DIST_TOL_M = 0.01
    _CUBE_REST_Z = 0.97

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create task objects (MUST use ONLY ColoredCube)
        self.cube_1 = ColoredCube(self._mojo)
        self.cube_2 = ColoredCube(self._mojo)
        self.cubes = [self.cube_1, self.cube_2]

        # REQUIRED: set_color() must be called in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() must be called in BOTH _initialize_env() AND _on_reset()
        self.cube_1.set_color("red")
        self.cube_2.set_color("blue")

        # 3. Place objects in RANDOM UNSOLVED positions on the table (within workspace bounds)
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = float(self._CUBE_REST_Z)

        # Ensure the initial configuration is NOT already successful:
        # - keep both on the table (z fixed)
        # - enforce distance != target within tolerance (with a small margin)
        target = float(self._TARGET_DISTANCE_M)
        tol = float(self._DIST_TOL_M)
        unsolved_margin = 0.01  # extra buffer so we don't accidentally start within tolerance
        min_sep = 0.06  # avoid starting overlapped/too-close (cube size dependent; conservative)

        def sample_point():
            return np.array(
                [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max), z],
                dtype=float,
            )

        pos1 = None
        pos2 = None
        for _ in range(200):
            p1 = sample_point()
            p2 = sample_point()
            d = float(np.linalg.norm(p1[:2] - p2[:2]))
            if d < min_sep:
                continue
            if abs(d - target) <= (tol + unsolved_margin):
                continue
            pos1, pos2 = p1, p2
            break

        # Fallback: guaranteed far-apart unsolved placement
        if pos1 is None or pos2 is None:
            pos1 = np.array([0.42, -0.22, z], dtype=float)
            pos2 = np.array([0.68, 0.22, z], dtype=float)

        # Random yaw rotations (still resting on table)
        def yaw_quat(yaw: float) -> np.ndarray:
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        q1 = yaw_quat(float(rng.uniform(-np.pi, np.pi)))
        q2 = yaw_quat(float(rng.uniform(-np.pi, np.pi)))

        self.cube_1.set_pose(position=pos1, quat=q1)
        self.cube_2.set_pose(position=pos2, quat=q2)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Goal checks ---
        # Condition A/B: both cubes resting on the table surface
        def _on_table(cube: ColoredCube) -> bool:
            # Collision with table top is the primary check; Z-range is a stability sanity check.
            z = float(cube.body.get_position()[2])
            z_ok = (self._CUBE_REST_Z - 0.02) <= z <= (self._CUBE_REST_Z + 0.02)
            return bool(cube.is_colliding(self.table.table_top) and z_ok)

        cube1_on_table = _on_table(self.cube_1)
        cube2_on_table = _on_table(self.cube_2)

        # Condition C: distance between centers is 10 cm within tolerance
        d = float(distance(self.cube_1.body, self.cube_2.body))
        dist_ok = abs(d - self._TARGET_DISTANCE_M) <= self._DIST_TOL_M

        positions_ok = bool(cube1_on_table and cube2_on_table and dist_ok)

        # No semantic constraints beyond spatial arrangement for this task
        semantic_ok = True

        # Released: robot not holding either cube in any gripper
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # --- Subgoal staging (best-effort, derived from current state) ---
        cube1_held = any(self.robot.is_gripper_holding_object(self.cube_1, g) for g in self.robot.grippers)
        cube2_held = any(self.robot.is_gripper_holding_object(self.cube_2, g) for g in self.robot.grippers)

        # sg1: grasp cube_1
        self._metric_stage(0, bool(cube1_held))

        # sg2: place cube_1 on table (and not held)
        self._metric_stage(1, bool(cube1_on_table and (not cube1_held)))

        # sg3: grasp cube_2
        self._metric_stage(2, bool(cube2_held))

        # sg4: adjust cube_2 so distance condition is satisfied while both on table
        self._metric_stage(3, bool(cube1_on_table and cube2_on_table and dist_ok))

        # sg5: release all (both cubes not held)
        self._metric_stage(4, bool(released))

        # --- Final success ---
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "target_m": float(self._TARGET_DISTANCE_M),
                "tolerance_m": float(self._DIST_TOL_M),
                "measured_m": float(d),
                "abs_error_m": float(abs(d - self._TARGET_DISTANCE_M)),
            },
            pose_error={
                "cube_1_z_error_m": float(abs(float(self.cube_1.body.get_position()[2]) - self._CUBE_REST_Z)),
                "cube_2_z_error_m": float(abs(float(self.cube_2.body.get_position()[2]) - self._CUBE_REST_Z)),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})