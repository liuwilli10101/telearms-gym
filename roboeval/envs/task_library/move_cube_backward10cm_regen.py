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

class CubeBackwardShift(RoboEvalEnv, MetricRolloutEval):
    """
    Move a single colored cube exactly 10 cm backward (away from the robot/camera)
    from its starting position. Backward is along +X in the workspace convention.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the colored cube.
        2. Lift the cube slightly to reduce table friction before translating.
        3. Translate the cube backward by moving along +X by about 10 cm.
        4. Place the cube down at the new location after the backward shift.
        5. Release the cube and retract safely.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal parameters
    _AXIS_INDEX = 0            # x
    _DELTA_METERS = 0.10       # +10 cm
    _TOL_METERS = 0.02         # ±2 cm
    _MIN_X = 0.35
    _MAX_X = 0.75
    _MIN_Y = -0.30
    _MAX_Y = 0.30
    _CUBE_Z_ON_TABLE = 0.97

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create required asset (ONLY ColoredCube)
        self.coloredcube = ColoredCube(self._mojo)
        self.coloredcube.set_color("blue")  # REQUIRED in _initialize_env

        # Keep a standard list for metric hooks
        self.cubes = [self.coloredcube]

        # Internal references (may be updated if an external randomizer relocates the cube)
        self._initial_pos = None
        self._goal_pos = None
        self._need_initial_update = True

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

    def goal_specification(self) -> Dict:
        return {
            "task-name": "cube-backward-shift",
            "task-instruction": "Move the colored cube 10 cm backward (away from the robot), i.e., increase its x-position by 0.10 m.",
            "goal-conditions": [
                {
                    "type": "position",
                    "subject": "colored_cube",
                    "parameters": {
                        "displacement_axis": "x",
                        "axis_index": 0,
                        "delta_meters": self._DELTA_METERS,
                        "direction_sign": "+",
                        "tolerance_meters": self._TOL_METERS,
                    },
                }
            ],
        }

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

        # REQUIRED in _on_reset
        self.coloredcube.set_color("blue")

        # 3. Place objects in RANDOM UNSOLVED positions (ensure _success() == False after reset)
        # Workspace bounds required by prompt:
        #   X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        #
        # Strategy:
        #   - Sample an "initial reference" pose within bounds.
        #   - Define a goal pose = initial + 0.10m along +X (may be outside the sampling bounds; that's OK).
        #   - Place the cube at a *different* pose that is NOT within tolerance of the goal displacement.
        init_x = float(rng.uniform(0.40, 0.70))
        init_y = float(rng.uniform(-0.25, 0.25))
        init_pos = np.array([init_x, init_y, self._CUBE_Z_ON_TABLE], dtype=float)

        goal_pos = init_pos.copy()
        goal_pos[self._AXIS_INDEX] += self._DELTA_METERS

        # Sample an unsolved cube pose within the same workspace bounds.
        # Reject poses that would accidentally satisfy the displacement condition at reset.
        # (i.e., delta_x within [0.08, 0.12] and non-negative)
        for _ in range(100):
            cur_x = float(rng.uniform(0.40, 0.70))
            cur_y = float(rng.uniform(-0.25, 0.25))
            cur_pos = np.array([cur_x, cur_y, self._CUBE_Z_ON_TABLE], dtype=float)
            delta_x = float(cur_pos[self._AXIS_INDEX] - init_pos[self._AXIS_INDEX])
            accidentally_solved = (abs(delta_x - self._DELTA_METERS) <= self._TOL_METERS) and (delta_x >= 0.0)
            if not accidentally_solved:
                break
        else:
            # Fallback: force an unsolved state by placing at the initial pose (delta_x == 0)
            cur_pos = init_pos.copy()

        self._initial_pos = init_pos
        self._goal_pos = goal_pos
        # We control placement here; do not allow wrapper-detection to overwrite initial/goal on first _success().
        self._need_initial_update = False

        # Place cube at the sampled unsolved pose
        self.coloredcube.set_pose(position=cur_pos, quat=np.array([0, 0, 0, 1], dtype=float))

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _maybe_update_initial_reference(self):
        """
        If an external randomization wrapper repositions the cube after reset,
        update the stored 'initial' pose to match the new episode start pose.
        """
        if not getattr(self, "_need_initial_update", False):
            return
        if self._goal_pos is None:
            return

        cur = self.coloredcube.body.get_position()

        # If the cube is no longer near the "goal pose" we placed at reset, assume randomization happened.
        if float(np.linalg.norm(cur - self._goal_pos)) > 0.03:
            self._initial_pos = cur.copy()
            self._goal_pos = cur.copy()
            self._goal_pos[self._AXIS_INDEX] += self._DELTA_METERS

        self._need_initial_update = False

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "coloredcube"):
            return False

        # Update initial reference if a wrapper randomized the cube pose after reset
        self._maybe_update_initial_reference()

        if self._initial_pos is None or self._goal_pos is None:
            return False

        cur_pos = self.coloredcube.body.get_position()
        delta_x = float(cur_pos[self._AXIS_INDEX] - self._initial_pos[self._AXIS_INDEX])

        # Goal condition: x increased by +0.10m (±0.02m); y and z unconstrained
        positions_ok = (abs(delta_x - self._DELTA_METERS) <= self._TOL_METERS) and (delta_x >= 0.0)

        # Released condition
        released = not any(
            self.robot.is_gripper_holding_object(self.coloredcube, g) for g in self.robot.grippers
        )

        # Stage heuristics (best-effort, single-arm)
        grasped = not released
        lifted = float(cur_pos[2]) > (self._CUBE_Z_ON_TABLE + 0.015)
        transported = delta_x >= (self._DELTA_METERS - self._TOL_METERS)
        on_table = self.coloredcube.is_colliding(self.table.table_top)

        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(transported and on_table))
        self._metric_stage(4, bool(transported and on_table and released))

        # No semantic content to verify in this task
        semantic_ok = True

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        x_err = float(cur_pos[self._AXIS_INDEX] - self._goal_pos[self._AXIS_INDEX])
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "colored_cube_x_to_goal_abs_m": abs(x_err),
                "colored_cube_delta_x_abs_error_m": abs(delta_x - self._DELTA_METERS),
            },
            pose_error={
                "colored_cube_x_error_m": x_err,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})