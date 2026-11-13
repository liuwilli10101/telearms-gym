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

class GreenOnYellowStack(RoboEvalEnv, MetricRolloutEval):
    """Stack green on top of yellow.

Success Criteria:
- The green cube is stacked on the yellow cube with contact, XY alignment, and the expected Z gap.
- The yellow base is on the table, colors are green/yellow, and no gripper is holding either cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Geometric constants for ColoredCube (4cm)
    _CUBE_SIZE = 0.04
    _TABLE_Z = 0.95
    _CUBE_REST_Z = 0.97
    _STACK_Z = _CUBE_REST_Z + _CUBE_SIZE  # 1.01

    # Success tolerances
    _XY_TOL = 0.014  # per task constraints
    _Z_TOL = 0.015

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create task objects (ONLY allowed asset: ColoredCube)
        self.green_cube = ColoredCube(self._mojo)
        self.yellow_cube = ColoredCube(self._mojo)
        self.cubes = [self.green_cube, self.yellow_cube]

        # REQUIRED: set_color() in _initialize_env()
        self.green_cube.set_color("green")
        self.yellow_cube.set_color("yellow")

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization (pipeline may perturb later)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in _on_reset()
        self.green_cube.set_color("green")
        self.yellow_cube.set_color("yellow")

        # 3. Place objects in RANDOM *unsolved* state within workspace bounds
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Sample two distinct XY positions with sufficient separation to avoid accidental stacking success
        min_sep = 0.08  # > success XY tolerance and comfortably separated
        for _ in range(200):
            yellow_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            green_xy = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            if np.linalg.norm(green_xy - yellow_xy) >= min_sep:
                break
        else:
            # Fallback: deterministic separated placement inside bounds
            yellow_xy = np.array([0.50, -0.10], dtype=float)
            green_xy = np.array([0.62, 0.10], dtype=float)

        # Random yaw (keep upright)
        def yaw_quat():
            yaw = rng.uniform(-np.pi, np.pi)
            return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=float)

        # Place both cubes on the table (not stacked) so _success() is False after reset
        self.yellow_cube.set_pose(position=np.array([yellow_xy[0], yellow_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        self.green_cube.set_pose(position=np.array([green_xy[0], green_xy[1], z], dtype=float), quat=yaw_quat())
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(6):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_released(self) -> bool:
        return not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

    def _is_green_on_yellow(self) -> bool:
        gp = self.green_cube.body.get_position()
        yp = self.yellow_cube.body.get_position()

        xy_close = np.linalg.norm(gp[:2] - yp[:2]) < self._XY_TOL
        z_expected = (yp[2] + self._CUBE_SIZE)
        z_close = abs(gp[2] - z_expected) < self._Z_TOL

        colliding = self.green_cube.is_colliding(self.yellow_cube)

        # Green should not be touching the table when stacked on yellow
        green_off_table = not self.green_cube.is_colliding(self.table.table_top)

        # Green must be above yellow (simple directional sanity check)
        above = gp[2] > yp[2] + 0.5 * self._CUBE_SIZE

        return bool(xy_close and z_close and colliding and green_off_table and above)

    def _is_yellow_on_table(self) -> bool:
        yp = self.yellow_cube.body.get_position()
        on_table_collision = self.yellow_cube.is_colliding(self.table.table_top)
        z_close = abs(yp[2] - self._CUBE_REST_Z) < 0.02
        return bool(on_table_collision and z_close)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Color verification (ColoredCube-specific)
        colors_ok = (self.green_cube.get_color() == "green") and (self.yellow_cube.get_color() == "yellow")

        stacked_ok = self._is_green_on_yellow()
        base_ok = self._is_yellow_on_table()
        released = self._is_released()

        positions_ok = bool(stacked_ok and base_ok)

        # --- Subgoal staging (best-effort, sequential semantics) ---
        green_held = any(self.robot.is_gripper_holding_object(self.green_cube, s) for s in self.robot.grippers)
        green_pos = self.green_cube.body.get_position()
        yellow_pos = self.yellow_cube.body.get_position()

        # 1) grasp
        self._metric_stage(0, bool(green_held))

        # 2) lift (clear of table)
        lifted = bool((not self.green_cube.is_colliding(self.table.table_top)) and (green_pos[2] > self._CUBE_REST_Z + 0.01))
        self._metric_stage(1, bool(lifted))

        # 3) transport (roughly above yellow in XY)
        transported = bool(np.linalg.norm(green_pos[:2] - yellow_pos[:2]) < 0.05 and green_pos[2] > yellow_pos[2] + 0.02)
        self._metric_stage(2, bool(transported))

        # 4) align (tight XY alignment)
        aligned = bool(np.linalg.norm(green_pos[:2] - yellow_pos[:2]) < 0.010)
        self._metric_stage(3, bool(aligned))

        # 5) place (contact + correct height)
        placed = bool(self.green_cube.is_colliding(self.yellow_cube) and abs(green_pos[2] - (yellow_pos[2] + self._CUBE_SIZE)) < self._Z_TOL)
        self._metric_stage(4, bool(placed))

        # 6) release (final)
        self._metric_stage(5, bool(positions_ok and released))

        # Final success
        self._success_check = bool(positions_ok and colors_ok and released)

        # Metrics finalize
        gp = self.green_cube.body.get_position()
        yp = self.yellow_cube.body.get_position()
        green_to_yellow = float(distance(self.green_cube.body, self.yellow_cube.body))

        xy_err = float(np.linalg.norm(gp[:2] - yp[:2]))
        z_err = float(abs(gp[2] - (yp[2] + self._CUBE_SIZE)))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={"green_to_yellow_center": green_to_yellow},
            pose_error={"stack_xy_error": xy_err, "stack_z_error": z_err},
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})