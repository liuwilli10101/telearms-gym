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

class RedRightOfBlueCube(RoboEvalEnv, MetricRolloutEval):
    """Place red to the right of blue.

Success Criteria:
- The red cube is right of the blue cube by the configured negative-Y separation.
- Both cubes are on the table and keep their expected red/blue colors.
- No gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task constants
    _MIN_SEPARATION_M = 0.03
    _CUBE_Z_ON_TABLE = 0.97

    # Workspace bounds (task spec; stricter than global)
    _X_BOUNDS = (0.40, 0.70)
    _Y_BOUNDS = (-0.25, 0.25)

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube allowed)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)
        self.cubes = [self.red_cube, self.blue_cube]

        # REQUIRED: set_color() in _initialize_env
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # Initialize metrics (FULL signature REQUIRED)
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

        # 2) Deterministic RNG (REQUIRED pattern)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 3) Place objects in RANDOM UNSOLVED positions (success should be False)
        # Sample random poses within bounds, rejecting any configuration that already satisfies
        # the goal condition (blue_y - red_y >= _MIN_SEPARATION_M).
        def rand_quat_z():
            yaw = float(rng.uniform(-np.pi, np.pi))
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        max_tries = 200
        for _ in range(max_tries):
            red_x = float(rng.uniform(*self._X_BOUNDS))
            red_y = float(rng.uniform(*self._Y_BOUNDS))
            blue_x = float(rng.uniform(*self._X_BOUNDS))
            blue_y = float(rng.uniform(*self._Y_BOUNDS))

            y_sep = blue_y - red_y
            # Ensure UNSOLVED: violate minimum separation in the correct direction.
            # (i.e., red is NOT sufficiently to the right of blue)
            if y_sep < self._MIN_SEPARATION_M:
                break
        else:
            # Fallback: deterministically unsolved (red left of blue or too close)
            red_x, red_y = 0.55, 0.10
            blue_x, blue_y = 0.55, 0.09  # y_sep = -0.01 < _MIN_SEPARATION_M

        self.red_cube.set_pose(
            position=np.array([red_x, red_y, self._CUBE_Z_ON_TABLE], dtype=float),
            quat=rand_quat_z(),
        )
        self.blue_cube.set_pose(
            position=np.array([blue_x, blue_y, self._CUBE_Z_ON_TABLE], dtype=float),
            quat=rand_quat_z(),
        )

        # 4) Physics settling
        self._mojo.physics.forward()

        # 5) Initialize subgoals to False
        for idx in range(5):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _success(self) -> bool:
        # RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() can be called during reset
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False
        if not hasattr(self, "red_cube") or not hasattr(self, "blue_cube"):
            return False

        red_pos = self.red_cube.body.get_position()
        blue_pos = self.blue_cube.body.get_position()

        # Semantic verification (colors identify task objects)
        semantic_ok = (self.red_cube.get_color() == "red") and (self.blue_cube.get_color() == "blue")

        # Check they are on the table surface (collision with table top is reliable)
        red_on_table = self.red_cube.is_colliding(self.table.table_top)
        blue_on_table = self.blue_cube.is_colliding(self.table.table_top)

        # Goal condition: right_of along Y-axis => red_y < blue_y with minimum separation
        y_sep = float(blue_pos[1] - red_pos[1])  # positive if red is to the right (-Y) of blue
        positions_ok = bool(red_on_table and blue_on_table and (y_sep >= self._MIN_SEPARATION_M))

        # Released check (generic gripper iteration; no HandSide)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in [self.red_cube, self.blue_cube]
            for g in self.robot.grippers
        )

        # Subgoal staging (best-effort signals from robot/object state)
        holding_red = any(self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers)
        lifted_red = bool(holding_red and (red_pos[2] > self._CUBE_Z_ON_TABLE + 0.02))
        transported_right = bool(holding_red and (y_sep >= self._MIN_SEPARATION_M) and (red_pos[2] > self._CUBE_Z_ON_TABLE + 0.01))
        placed = bool(positions_ok and red_on_table)
        released_after_place = bool(placed and released)

        self._metric_stage(0, bool(holding_red))
        self._metric_stage(1, bool(lifted_red))
        self._metric_stage(2, bool(transported_right))
        self._metric_stage(3, bool(placed))
        self._metric_stage(4, bool(released_after_place))

        # Final success
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metric finalize (MUST be last metric call before return)
        y_sep_error = float(max(0.0, self._MIN_SEPARATION_M - y_sep))
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "y_separation_error_m": y_sep_error,
                "y_separation_m": float(y_sep),
            },
            pose_error={
                "red_y_minus_blue_y_m": float(red_pos[1] - blue_pos[1]),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})