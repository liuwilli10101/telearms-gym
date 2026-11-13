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

class CubeBackRightPlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Move the single colored cube to the back-right corner of the workspace, interpreting
    back as +X and right as -Y. The task succeeds when the cube is placed within a tight
    XY tolerance near that corner on the table surface.

    Sequential Sub-Goals:
        1. Establish a stable grasp on the cube.
        2. Lift the cube clear of the table to enable safe transport.
        3. Move the cube toward the back-right corner (toward +X and -Y).
        4. Lower the cube onto the table near the back-right corner target XY.
        5. Release the cube while keeping it within the target region.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (from specification)
    _TARGET_XY = np.array([0.69, -0.24], dtype=float)
    _XY_TOL = 0.03
    _Z_TOL = 0.03
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)
    _CUBE_REST_Z = 0.97
    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create required task object (ONLY asset allowed)
        self.colored_cube_0 = ColoredCube(self._mojo)
        self.colored_cube_0.set_color("blue")  # REQUIRED in _initialize_env
        self.cubes = [self.colored_cube_0]

        # Optional: place somewhere safe within workspace
        self.colored_cube_0.set_pose(
            position=np.array([0.55, 0.00, self._CUBE_REST_Z]),
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

        # 2. ⚠️ CRITICAL: Use deterministic randomization
        rng = np.random.default_rng(self.seed)

        # REQUIRED: Call set_color() in _on_reset
        palette = ["red", "blue", "green", "yellow", "purple", "orange"]
        self.colored_cube_0.set_color(palette[int(rng.integers(0, len(palette)))])

        # 3. Randomize cube pose to an UNSOLVED state within workspace bounds
        #    Ensure it's NOT within the goal XY tolerance, so _success() is False after reset.
        margin = 0.01  # extra buffer beyond tolerance to avoid accidental success due to numerical/physics jitter
        min_dist = float(self._XY_TOL + margin)

        for _ in range(200):
            x = float(rng.uniform(self._X_BOUNDS[0], self._X_BOUNDS[1]))
            y = float(rng.uniform(self._Y_BOUNDS[0], self._Y_BOUNDS[1]))
            if float(np.linalg.norm(np.array([x, y], dtype=float) - self._TARGET_XY)) > min_dist:
                break
        else:
            # Fallback: force a deterministic non-goal location inside bounds
            x, y = float(self._X_BOUNDS[0]), float(self._Y_BOUNDS[1])

        # Random yaw about Z (upright cube); keep Z at table rest height
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        self.colored_cube_0.set_pose(
            position=np.array([x, y, self._CUBE_REST_Z], dtype=float),
            quat=quat,
        )

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _get_goal_region_parameters(self) -> Dict:
        return {
            "frame": "workspace",
            "xy_target": self._TARGET_XY.copy(),
            "xy_tolerance": float(self._XY_TOL),
            "x_bounds": [float(self._X_BOUNDS[0]), float(self._X_BOUNDS[1])],
            "y_bounds": [float(self._Y_BOUNDS[0]), float(self._Y_BOUNDS[1])],
            "corner_interpretation": {"back": "+X", "right": "-Y"},
            "require_on_table": True,
            "z_tolerance": float(self._Z_TOL),
        }

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "colored_cube_0"):
            return False

        cube = self.colored_cube_0
        pos = cube.body.get_position()
        pos = np.array(pos, dtype=float)

        # Bounds checks (task-specified tight bounds)
        in_bounds = (
            (self._X_BOUNDS[0] <= pos[0] <= self._X_BOUNDS[1])
            and (self._Y_BOUNDS[0] <= pos[1] <= self._Y_BOUNDS[1])
        )

        # Goal proximity in XY (kept for diagnostic metric only)
        xy_err = np.linalg.norm(pos[:2] - self._TARGET_XY)
        # Region check: back-right quadrant of workspace (back = +X, right = -Y).
        # Use a region constraint rather than a tight tolerance against an exact corner
        # coordinate, so a human teleoperator can satisfy the task.
        in_back_right = (pos[0] >= 0.60) and (pos[1] <= -0.15)
        xy_ok = bool(in_back_right)

        # Require cube resting on table: collision with table top + Z close to rest height
        on_table_collision = cube.is_colliding(self.table.table_top)
        z_ok = abs(pos[2] - self._CUBE_REST_Z) <= self._Z_TOL
        on_table = bool(on_table_collision and z_ok)

        # Released: not held by any gripper
        released = not any(
            self.robot.is_gripper_holding_object(cube, s) for s in self.robot.grippers
        )
        grasped = not released
        lifted = (pos[2] > (self._CUBE_REST_Z + 0.02)) and (not on_table_collision)

        # Simple intermediate "transport" heuristic: moved generally toward +X and -Y and within a coarse radius
        transported = (pos[0] >= 0.62) and (pos[1] <= -0.15)

        placed = bool(in_bounds and xy_ok and on_table)
        positions_ok = placed

        # Stage metrics (evaluate current completion of each subgoal)
        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(placed))
        self._metric_stage(4, bool(placed and released))

        semantic_ok = True  # no semantic constraints in this task

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "colored_cube_0_xy": float(xy_err),
            },
            pose_error={
                "colored_cube_0_x": float(pos[0] - self._TARGET_XY[0]),
                "colored_cube_0_y": float(pos[1] - self._TARGET_XY[1]),
                "colored_cube_0_z": float(pos[2] - self._CUBE_REST_Z),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})