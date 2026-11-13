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

class CubeRightShiftTask(RoboEvalEnv, MetricRolloutEval):
    """
    Move the single colored cube exactly 15 cm to the right of where it started.
    'Right' is defined as decreasing Y (workspace -Y direction).

    Sequential Sub-Goals:
        1. Obtain a stable grasp on the cube.
        2. Lift the cube slightly to clear the table for transport.
        3. Translate the cube rightward along -Y while keeping X roughly constant.
        4. Place the cube down at the right-shifted target location.
        5. Release the cube after placement.
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal parameters (axis_index=1 => Y)
    _DELTA_Y_M = -0.15
    _Y_TOL_M = 0.02
    _X_DRIFT_TOL_M = 0.03
    _Z_DRIFT_TOL_M = 0.03

    _CUBE_REST_Z = 0.97

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Assets (MUST be ONLY ColoredCube) ---
        self.coloredcube = ColoredCube(self._mojo)
        self.coloredcube.set_color("blue")  # REQUIRED in _initialize_env

        # Standardize object list name used by metric init
        self.cubes = [self.coloredcube]

        # Stored per-episode reference pose (initial pose) and target pose
        self._initial_pos = None
        self._target_pos = None

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
        # REQUIRED in _on_reset
        self.coloredcube.set_color("blue")

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

        # 3. Randomize an UNSOLVED initial pose within workspace bounds,
        #    and store initial/target for success check.
        #
        # Workspace bounds required: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        # Also ensure the *target* (y0 + delta) stays within bounds.
        x0 = float(rng.uniform(0.40, 0.70))
        y0 = float(rng.uniform(-0.10, 0.25))  # ensures y0 + (-0.15) >= -0.25
        z0 = float(self._CUBE_REST_Z)

        self._initial_pos = np.array([x0, y0, z0], dtype=float)
        self._target_pos = np.array([x0, y0 + self._DELTA_Y_M, z0], dtype=float)  # right shift => -Y

        # Place cube at the INITIAL (unsolved) location (not at target).
        # Add small random XY jitter that keeps it in-bounds and not accidentally "solved".
        max_jitter = 0.04  # > Y_TOL (0.02) to avoid immediate success
        jx = float(rng.uniform(-max_jitter, max_jitter))
        jy = float(rng.uniform(-max_jitter, max_jitter))

        pos = self._initial_pos.copy()
        pos[0] = float(np.clip(pos[0] + jx, 0.40, 0.70))
        pos[1] = float(np.clip(pos[1] + jy, -0.25, 0.25))
        pos[2] = z0

        # Ensure not at the target within tolerances after jitter (guarantee _success() is False).
        # If too close to target Y, push away along Y while staying in bounds.
        if abs(pos[1] - self._target_pos[1]) <= (self._Y_TOL_M + 1e-6):
            if pos[1] <= self._target_pos[1]:
                pos[1] = float(np.clip(self._target_pos[1] - (self._Y_TOL_M + 0.03), -0.25, 0.25))
            else:
                pos[1] = float(np.clip(self._target_pos[1] + (self._Y_TOL_M + 0.03), -0.25, 0.25))

        # Also avoid accidental X/Z success due to clipping: keep X within drift tol from initial
        # (success requires small drift, but we primarily ensure Y isn't solved; this keeps things sane).
        pos[0] = float(np.clip(pos[0], 0.40, 0.70))

        self.coloredcube.set_pose(
            position=pos,
            quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
        )

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

        # Defensive checks - _success() is called during reset!
        if (
            not hasattr(self, "cubes")
            or not hasattr(self, "table")
            or self._initial_pos is None
            or self._target_pos is None
        ):
            return False

        pos = self.coloredcube.body.get_position()

        # --- Goal condition: relative to initial reference pose along Y (axis_index=1) ---
        y_err = float(pos[1] - (self._initial_pos[1] + self._DELTA_Y_M))  # should be ~0
        x_drift = float(pos[0] - self._initial_pos[0])
        z_drift = float(pos[2] - self._initial_pos[2])

        positions_ok = (
            abs(y_err) <= self._Y_TOL_M
            and abs(x_drift) <= self._X_DRIFT_TOL_M
            and abs(z_drift) <= self._Z_DRIFT_TOL_M
        )

        # No semantic constraints for ColoredCube beyond task geometry
        semantic_ok = True

        # Must not be held
        released = not any(
            self.robot.is_gripper_holding_object(self.coloredcube, s) for s in self.robot.grippers
        )

        # Subgoal staging (atomic task; mark all once goal is satisfied)
        if positions_ok:
            for i in range(5):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "abs_y_error_m": abs(y_err),
                "abs_x_drift_m": abs(x_drift),
                "abs_z_drift_m": abs(z_drift),
            },
            pose_error={
                "y_error_m": y_err,
                "x_drift_m": x_drift,
                "z_drift_m": z_drift,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})