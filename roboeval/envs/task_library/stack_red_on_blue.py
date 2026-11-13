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

class RedOnBlueStacking(RoboEvalEnv, MetricRolloutEval):
    """Stack red on top of blue.

Success Criteria:
- The red cube is on the blue cube, aligned in XY, and separated by the expected stack height.
- The red cube is not touching the table, the blue cube is on the table, and no gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Spatial tolerances / constants for ColoredCube (4cm)
    _CUBE_SIZE = 0.04
    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97
    _STACK_Z = 0.97 + 0.04  # 1.01
    _XY_ALIGN_TOL = 0.014  # from task constraints
    _Z_STACK_TOL = 0.010

    _NUM_SUBGOALS = 7

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create task objects (MUST use ONLY ColoredCube)
        self.red_cube = ColoredCube(self._mojo)
        self.blue_cube = ColoredCube(self._mojo)

        # Trackable list
        self.cubes = [self.red_cube, self.blue_cube]

        # REQUIRED: set_color() in _initialize_env()
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

        # 2. Deterministic RNG (required pattern; randomization may be applied externally)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() in _on_reset()
        self.red_cube.set_color("red")
        self.blue_cube.set_color("blue")

        # 4. RANDOM UNSOLVED placement within workspace bounds
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z0 = self._CUBE_REST_Z

        def rand_xy():
            return np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)

        # Sample two positions that are not stacked/aligned (ensure success() is False)
        # Keep them well-separated in XY to avoid accidental "on" relation or alignment.
        min_xy_sep = max(0.08, self._XY_ALIGN_TOL * 4.0)  # comfortably > alignment tolerance
        blue_xy = rand_xy()
        red_xy = rand_xy()
        for _ in range(200):
            if float(np.linalg.norm(red_xy - blue_xy)) >= min_xy_sep:
                break
            red_xy = rand_xy()

        # Random yaw rotations (quat as [x,y,z,w])
        blue_yaw = float(rng.uniform(-np.pi, np.pi))
        red_yaw = float(rng.uniform(-np.pi, np.pi))
        blue_quat = np.array([0.0, 0.0, np.sin(blue_yaw / 2.0), np.cos(blue_yaw / 2.0)], dtype=float)
        red_quat = np.array([0.0, 0.0, np.sin(red_yaw / 2.0), np.cos(red_yaw / 2.0)], dtype=float)

        # Place cubes on table (not stacked) and settle
        self.blue_cube.set_pose(position=np.array([blue_xy[0], blue_xy[1], z0], dtype=float), quat=blue_quat)
        self.red_cube.set_pose(position=np.array([red_xy[0], red_xy[1], z0], dtype=float), quat=red_quat)

        for _i in range(80):
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
        if not hasattr(self, "red_cube") or not hasattr(self, "blue_cube"):
            return False

        red_p = self.red_cube.body.get_position()
        blue_p = self.blue_cube.body.get_position()

        # --- Goal checks: "on" + "aligned" ---
        xy_dist = float(np.linalg.norm((red_p - blue_p)[:2]))
        z_gap = float(red_p[2] - blue_p[2])
        z_gap_err = float(abs(z_gap - self._CUBE_SIZE))

        aligned = xy_dist <= self._XY_ALIGN_TOL
        on_relation = (
            self.red_cube.is_colliding(self.blue_cube)
            and (not self.red_cube.is_colliding(self.table.table_top))
        )

        # Encourage physically sensible stack: blue on table, correct height gap
        blue_on_table = self.blue_cube.is_colliding(self.table.table_top)
        z_ok = z_gap_err <= self._Z_STACK_TOL

        positions_ok = bool(on_relation and aligned and z_ok and blue_on_table)

        # Released: robot not holding either cube
        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # --- Subgoal staging (best-effort, single-arm safe) ---
        # 0: blue stable (on table, not held)
        blue_not_held = not any(
            self.robot.is_gripper_holding_object(self.blue_cube, g) for g in self.robot.grippers
        )
        if blue_on_table and blue_not_held:
            self._metric_stage(0, True)

        # 1: grasp red
        red_held = any(
            self.robot.is_gripper_holding_object(self.red_cube, g) for g in self.robot.grippers
        )
        if red_held:
            self._metric_stage(1, True)

        # 2: lift red (not touching table)
        red_off_table = not self.red_cube.is_colliding(self.table.table_top)
        if red_held and red_off_table:
            self._metric_stage(2, True)

        # 3: transport above blue (roughly above in XY and higher than blue)
        above_blue = (xy_dist <= 0.05) and (red_p[2] >= blue_p[2] + 0.02)
        if red_held and above_blue:
            self._metric_stage(3, True)

        # 4: align (tight XY)
        if red_held and aligned:
            self._metric_stage(4, True)

        # 5: place (supported by blue, correct z gap)
        if on_relation and z_ok:
            self._metric_stage(5, True)

        # 6: release after stable stacking
        if positions_ok and released:
            self._metric_stage(6, True)

        # Final success condition
        self._success_check = bool(positions_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "red_to_blue_xy": xy_dist,
                "red_to_blue_zgap_err": z_gap_err,
            },
            pose_error={
                "xy_align_error": xy_dist,
                "z_gap_error": z_gap_err,
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})