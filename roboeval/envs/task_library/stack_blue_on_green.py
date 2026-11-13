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

class BlueOnGreenCubeStacking(RoboEvalEnv, MetricRolloutEval):
    """Stack the blue cube on top of the green cube.

Success Criteria:
- The blue cube contacts the green cube, is not on the table, and is XY/Z aligned as the top cube.
- The green cube color is green, the blue cube color is blue, and no gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _CUBE_REST_Z = 0.97
    _CUBE_SIZE = 0.04
    _STACK_DZ = 0.04

    _ALIGN_TOL_XY = 0.014  # task constraint tolerance
    _HEIGHT_TOL = 0.015

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ColoredCube only)
        self.green_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")

        self.cubes = [self.green_cube, self.blue_cube]

        # Place somewhere safe initially (will be overwritten in _on_reset)
        self.green_cube.set_pose(
            position=np.array([0.60, 0.10, self._CUBE_REST_Z]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
        )
        self.blue_cube.set_pose(
            position=np.array([0.60, -0.10, self._CUBE_REST_Z]),
            quat=np.array([0.0, 0.0, 0.0, 1.0]),
        )

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

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.green_cube.set_color("green")
        self.blue_cube.set_color("blue")

        # 2. ⚠️ CRITICAL: Use deterministic randomization (pipeline may perturb later)
        rng = np.random.default_rng(self.seed)

        # 3. Randomize initial poses in an UNSOLVED configuration within workspace bounds
        x_low, x_high = 0.40, 0.70
        y_low, y_high = -0.25, 0.25
        z0 = self._CUBE_REST_Z

        # Ensure cubes are not stacked/aligned: enforce a minimum XY separation
        # (well above alignment tolerance and cube size)
        min_xy_sep = 0.12

        def rand_xy():
            return np.array([rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)], dtype=float)

        green_xy = rand_xy()
        blue_xy = rand_xy()
        tries = 0
        while float(np.linalg.norm(blue_xy - green_xy)) < min_xy_sep and tries < 200:
            blue_xy = rand_xy()
            tries += 1
        if float(np.linalg.norm(blue_xy - green_xy)) < min_xy_sep:
            # Deterministic fallback far apart in XY (still within bounds)
            green_xy = np.array([0.45, 0.20], dtype=float)
            blue_xy = np.array([0.65, -0.20], dtype=float)

        green_pos = np.array([green_xy[0], green_xy[1], z0], dtype=float)
        blue_pos = np.array([blue_xy[0], blue_xy[1], z0], dtype=float)

        # Random yaw-only orientations (stable on table)
        def yaw_quat(yaw: float) -> np.ndarray:
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        green_quat = yaw_quat(rng.uniform(-np.pi, np.pi))
        blue_quat = yaw_quat(rng.uniform(-np.pi, np.pi))

        # 4. Layer-by-layer placement for physics stability
        self.green_cube.set_pose(position=green_pos, quat=green_quat)
        self._mojo.physics.forward()

        self.blue_cube.set_pose(position=blue_pos, quat=blue_quat)
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
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
        if not hasattr(self, "green_cube") or not hasattr(self, "blue_cube"):
            return False

        blue_p = self.blue_cube.body.get_position()
        green_p = self.green_cube.body.get_position()

        # --- Goal checks: "blue on green" + "aligned/centered" + "released" ---
        # On-relation: collision AND blue not on table
        stacked_contact = self.blue_cube.is_colliding(self.green_cube)
        not_on_table = not self.blue_cube.is_colliding(self.table.table_top)

        # Alignment: XY centers close + expected height difference
        dxy = float(np.linalg.norm((blue_p - green_p)[:2]))
        dz_err = float(abs((blue_p[2] - green_p[2]) - self._STACK_DZ))
        aligned = (dxy < self._ALIGN_TOL_XY) and (dz_err < self._HEIGHT_TOL)

        # Color sanity (ColoredCube-only task)
        colors_ok = (self.blue_cube.get_color() == "blue") and (self.green_cube.get_color() == "green")

        # Released: no gripper holding either cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        positions_ok = bool(stacked_contact and not_on_table and aligned)

        # --- Subgoal staging (best-effort signals) ---
        holding_blue = any(
            self.robot.is_gripper_holding_object(self.blue_cube, s) for s in self.robot.grippers
        )
        lifted_blue = (blue_p[2] > (self._CUBE_REST_Z + 0.02)) and (not self.blue_cube.is_colliding(self.table.table_top))
        above_green = (dxy < 0.05) and (blue_p[2] > green_p[2] + 0.06)
        aligned_above = (dxy < self._ALIGN_TOL_XY) and (blue_p[2] > green_p[2] + 0.03)
        placed_on_green = bool(stacked_contact and aligned and not_on_table)

        self._metric_stage(0, bool(holding_blue))
        self._metric_stage(1, bool(lifted_blue))
        self._metric_stage(2, bool(above_green))
        self._metric_stage(3, bool(aligned_above))
        self._metric_stage(4, bool(placed_on_green))
        self._metric_stage(5, bool(placed_on_green and released))

        self._success_check = bool(positions_ok and colors_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "blue_to_green_xy": dxy,
                "blue_to_green_stack_dz_err": dz_err,
            },
            pose_error={
                "blue_alignment_xy_err": dxy,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})