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

class CubePositionSwapTask(RoboEvalEnv, MetricRolloutEval):
    """
    Two colored cubes (red and blue) start at fixed table positions A and B.
    The robot must swap them so the red cube ends at B and the blue cube ends at A.

    Sequential Sub-Goals:
        1. Grasp the red cube with the left gripper (or any gripper).
        2. Lift the red cube clear of the table to enable safe transport.
        3. Move the red cube toward position B.
        4. Place the red cube down at position B.
        5. Release the red cube after placing at position B.
        6. Grasp the blue cube with the right gripper (or any gripper).
        7. Lift the blue cube clear of the table to enable safe transport.
        8. Move the blue cube toward position A.
        9. Place the blue cube down at position A.
        10. Release the blue cube after placing at position A.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Goal tolerance from task spec
    _TOL_M = 0.03
    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Fixed reference positions (A and B) in workspace bounds
        self._pos_A = np.array([0.55, 0.12, 0.97])   # A (left side, +Y)
        self._pos_B = np.array([0.55, -0.12, 0.97])  # B (right side, -Y)
        self._quat_upright = np.array([0.0, 0.0, 0.0, 1.0])  # xyzw

        # Create assets (ONLY ColoredCube allowed)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        self.cubes = [self.red_cube, self.blue_cube]

        # Initialize metrics (full signature required)
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

        # 2. Deterministic RNG (required pattern; randomization added by pipeline later)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM *unsolved* positions so _success() returns False
        # REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z_base = 0.97

        # Keep objects well-separated and away from goal states
        min_sep = 0.06
        goal_clear = float(self._TOL_M) + 0.01  # ensure not within success tolerance

        def sample_pos():
            return np.array(
                [
                    rng.uniform(x_min, x_max),
                    rng.uniform(y_min, y_max),
                    z_base,
                ],
                dtype=float,
            )

        red_pos = None
        blue_pos = None
        for _ in range(1000):
            rp = sample_pos()
            bp = sample_pos()

            # Keep within bounds (already), avoid overlap
            if np.linalg.norm(rp - bp) < min_sep:
                continue

            # Ensure not already in goal configuration (red at B AND blue at A)
            if np.linalg.norm(rp - self._pos_B) <= self._TOL_M and np.linalg.norm(bp - self._pos_A) <= self._TOL_M:
                continue

            # Ensure at least one is clearly not at its goal (practically always, but enforce)
            if np.linalg.norm(rp - self._pos_B) <= goal_clear and np.linalg.norm(bp - self._pos_A) <= goal_clear:
                continue

            red_pos, blue_pos = rp, bp
            break

        # Fallback (should be extremely unlikely): force an unsolved arrangement
        if red_pos is None or blue_pos is None:
            red_pos = self._pos_A.copy()
            blue_pos = self._pos_B.copy()

        self.red_cube.set_pose(position=red_pos.copy(), quat=self._quat_upright.copy())
        self.blue_cube.set_pose(position=blue_pos.copy(), quat=self._quat_upright.copy())

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
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
        if not hasattr(self, "_pos_A") or not hasattr(self, "_pos_B"):
            return False

        # Goal conditions: red at B, blue at A (within tolerance)
        red_pos = np.array(self.red_cube.body.get_position())
        blue_pos = np.array(self.blue_cube.body.get_position())

        red_to_B = float(np.linalg.norm(red_pos - self._pos_B))
        blue_to_A = float(np.linalg.norm(blue_pos - self._pos_A))

        red_at_B = red_to_B <= self._TOL_M
        blue_at_A = blue_to_A <= self._TOL_M
        positions_ok = bool(red_at_B and blue_at_A)

        # Semantic/color check (defensive; colors should remain correct)
        semantic_ok = bool(self.red_cube.get_color() == "red" and self.blue_cube.get_color() == "blue")

        # Released (no cube held by any gripper)
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        released_red = not any(
            self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers
        )
        released_blue = not any(
            self.robot.is_gripper_holding_object(self.blue_cube, g) for g in self.robot.grippers
        )

        # Subgoal staging (best-effort from final state signals)
        # Red cube sequence
        if red_at_B:
            for i in range(0, 4):  # grasp/lift/transport/place (approx as achieved if at goal)
                self._metric_stage(i, True)
            if released_red:
                self._metric_stage(4, True)

        # Blue cube sequence
        if blue_at_A:
            for i in range(5, 9):  # grasp/lift/transport/place (approx as achieved if at goal)
                self._metric_stage(i, True)
            if released_blue:
                self._metric_stage(9, True)

        # If fully successful, mark all stages as complete
        if positions_ok and semantic_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        # Set success flag
        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_position_B_m": red_to_B,
                "blue_to_position_A_m": blue_to_A,
            },
            pose_error={
                "red_position_error_m": red_to_B,
                "blue_position_error_m": blue_to_A,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})