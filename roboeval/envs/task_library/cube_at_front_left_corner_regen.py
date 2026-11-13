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

class PlaceCubeFrontLeft(RoboEvalEnv, MetricRolloutEval):
    """
    Move the single colored cube to the front-left corner of the workspace.

    Axis convention:
        -X = front, +X = back
        +Y = left,  -Y = right
        +Z = up

    Goal:
        Place the cube at (x=0.40, y=0.25) within xy tolerance 0.03.

    Sequential Sub-Goals:
        1. Grasp the cube with either gripper.
        2. Lift the cube clear of the table.
        3. Transport the cube toward the front-left target.
        4. Place the cube down at the target location.
        5. Release the cube after placement.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task parameters (per specification)
    _WORKSPACE_BOUNDS = {"x": (0.40, 0.70), "y": (-0.25, 0.25)}
    _TARGET_XY = np.array([0.40, 0.25], dtype=float)
    _XY_TOL = 0.03
    _CUBE_REST_Z = 0.97
    _NUM_SUBGOALS = 5

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create props (ONLY allowed asset: ColoredCube)
        self.colored_cube = ColoredCube(self._mojo)
        self.colored_cube.set_color("blue")  # REQUIRED in _initialize_env

        # Convenience list for metrics / generic loops
        self.cubes = [self.colored_cube]

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

        # 2. Deterministic RNG handle (required pattern; randomization may be applied externally)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM UNSOLVED positions (ensure not within goal tolerance)
        self.colored_cube.set_color("blue")  # REQUIRED in _on_reset

        x_low, x_high = self._WORKSPACE_BOUNDS["x"]
        y_low, y_high = self._WORKSPACE_BOUNDS["y"]

        # Sample until definitely outside the success region (XY distance > tolerance)
        min_dist = float(self._XY_TOL) + 1e-3
        for _ in range(1000):
            x = float(rng.uniform(x_low, x_high))
            y = float(rng.uniform(y_low, y_high))
            if float(np.linalg.norm(np.array([x, y], dtype=float) - self._TARGET_XY)) > min_dist:
                break
        else:
            # Fallback: place at a corner far from target within bounds
            x, y = float(x_high), float(y_low)

        # Random yaw about Z to vary orientation while keeping cube on table
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        init_pos = np.array([x, y, self._CUBE_REST_Z], dtype=float)
        self.colored_cube.set_pose(position=init_pos, quat=quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

        for _ in range(1000):
            x = float(rng.uniform(x_low, x_high))
            y = float(rng.uniform(y_low, y_high))
            if float(np.linalg.norm(np.array([x, y], dtype=float) - self._TARGET_XY)) > min_dist:
                break
        else:
            # Fallback: place at a corner far from target within bounds
            x, y = float(x_high), float(y_low)

        # Random yaw about Z to vary orientation while keep
    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "colored_cube"):
            return False

        cube = self.colored_cube
        pos = cube.body.get_position()

        # --- Core goal check: cube placed in the front-left region of the workspace ---
        # Front = -X (low X), left = +Y (high Y). Use a region constraint instead of a
        # tight tolerance against the exact corner so a human teleoperator can succeed.
        xy_dist = float(np.linalg.norm(pos[:2] - self._TARGET_XY))  # diagnostic only
        in_front_left = (pos[0] <= 0.50) and (pos[1] >= 0.15)
        positions_ok = bool(in_front_left)

        # Ensure cube is on the table surface (not airborne / not dropped elsewhere)
        on_table = cube.is_colliding(self.table.table_top)

        # Released: no gripper is holding the cube
        released = not any(
            self.robot.is_gripper_holding_object(cube, g) for g in getattr(self.robot, "grippers", [])
        )

        # --- Subgoal heuristics (best-effort for metrics staging) ---
        # 1) grasp
        grasped = any(
            self.robot.is_gripper_holding_object(cube, g) for g in getattr(self.robot, "grippers", [])
        )
        # 2) lift (cube clearly above resting height)
        lifted = bool(pos[2] > (self._CUBE_REST_Z + 0.02))
        # 3) transport (near target while elevated)
        transported = bool((xy_dist <= 2.0 * self._XY_TOL) and lifted)
        # 4) place (near target and on table)
        placed = bool(positions_ok and on_table)
        # 5) release (placed and released)
        released_after_place = bool(placed and released)

        self._metric_stage(0, bool(grasped))
        self._metric_stage(1, bool(lifted))
        self._metric_stage(2, bool(transported))
        self._metric_stage(3, bool(placed))
        self._metric_stage(4, bool(released_after_place))

        semantic_ok = True  # no semantic constraints for ColoredCube in this task

        self._success_check = bool(positions_ok and on_table and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"colored_cube_0_xy": xy_dist},
            pose_error={
                "colored_cube_0_dx": float(pos[0] - self._TARGET_XY[0]),
                "colored_cube_0_dy": float(pos[1] - self._TARGET_XY[1]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})