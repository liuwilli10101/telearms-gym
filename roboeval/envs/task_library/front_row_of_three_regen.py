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


class FrontEdgeCubeRow(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange three colored cubes into a single horizontal row along the workspace front edge
    closest to the robot. The row only needs to sit anywhere within a band near the front
    edge (a range of x, not a precise line); the three cubes just need to share roughly the
    same x so it reads as one row, and be spread apart along y.

    Sequential Sub-Goals:
        1. Grasp the red cube with either gripper.
        2. Move the red cube toward the front edge of the workspace.
        3. Place the red cube down on the table near the front edge to start the row.
        4. Grasp the green cube with either gripper.
        5. Move the green cube toward the front edge of the workspace.
        6. Place the green cube down near the front edge, aligned with the first cube.
        7. Grasp the blue cube with either gripper.
        8. Move the blue cube toward the front edge of the workspace.
        9. Place the blue cube down near the front edge to complete the row.
        10. Fine-adjust so all three cubes share roughly the same x near the front edge and
            are spaced apart along y to form a clean row.
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # --- Task constants (from specification) ---
    # The real table front edge near the robot sits around x ~ 0.32 (measured from teleop:
    # cubes pushed to the front rest at x ~ 0.322), NOT x = 0.40. The row may sit anywhere
    # within a band near that real front edge, not on a precise line.
    _X_TARGET = 0.32            # front-edge reference (real table front edge near the robot)
    _FRONT_X_MIN = 0.29        # front-edge band: row may sit anywhere in [0.29, 0.45]
    _FRONT_X_MAX = 0.45
    _ROW_X_SPREAD_TOL = 0.03    # the three cubes' x must agree within this to read as one row

    _MIN_PAIRWISE_Y_SEP = 0.05
    _MAX_PAIRWISE_Y_SEP = 0.28

    _CUBE_Z_ON_TABLE = 0.97
    _UP_AXIS_COS_TOL = np.cos(np.deg2rad(30.0))  # "upright" tolerance (cube is symmetric; keep it loose)

    _NUM_SUBGOALS = 10

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # --- Objects (ONLY ColoredCube allowed) ---
        self.cube_red = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        self.cubes = [self.cube_red, self.cube_green, self.cube_blue]
        self._cube_names = ["cube_red", "cube_green", "cube_blue"]

        # A deterministic goal row (success condition allows any y ordering as long as separated)
        self._goal_positions = {
            "cube_red": np.array([self._X_TARGET, 0.08, self._CUBE_Z_ON_TABLE]),
            "cube_green": np.array([self._X_TARGET, 0.00, self._CUBE_Z_ON_TABLE]),
            "cube_blue": np.array([self._X_TARGET, -0.08, self._CUBE_Z_ON_TABLE]),
        }
        self._goal_quat = np.array([0.0, 0.0, 0.0, 1.0])

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

        # 2. ⚠️ CRITICAL: Use deterministic randomization handle (even if not used here)
        rng = np.random.default_rng(self.seed)

        # 3. Randomize object poses to UNSOLVED positions within workspace bounds
        self.cube_red.set_color("red")
        self.cube_green.set_color("green")
        self.cube_blue.set_color("blue")

        # Workspace bounds
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z = self._CUBE_Z_ON_TABLE

        # Ensure reset is NOT already successful: spawn cubes BEHIND the front band so
        # at least the front-band check fails (x > _FRONT_X_MAX + margin).
        sample_x_min = max(x_min, float(self._FRONT_X_MAX) + 0.03)

        # Also avoid overlaps by enforcing a minimum XY separation
        min_xy_sep = 0.07

        positions = []
        for _ in range(3):
            for _attempt in range(200):
                x = float(rng.uniform(sample_x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y, z], dtype=float)

                if all(np.linalg.norm(p[:2] - q[:2]) >= min_xy_sep for q in positions):
                    positions.append(p)
                    break
            else:
                # Fallback: deterministic spaced placement still off the target line (unsolved)
                idx = len(positions)
                positions.append(
                    np.array(
                        [
                            float(sample_x_min),
                            float(np.clip(-0.20 + 0.20 * idx, y_min, y_max)),
                            z,
                        ],
                        dtype=float,
                    )
                )

        self.cube_red.set_pose(position=positions[0], quat=self._goal_quat)
        self.cube_green.set_pose(position=positions[1], quat=self._goal_quat)
        self.cube_blue.set_pose(position=positions[2], quat=self._goal_quat)

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_upright(self, cube: ColoredCube) -> bool:
        # mojo get_quaternion() returns MuJoCo wxyz order; pyquaternion also expects (w,x,y,z).
        q_wxyz = cube.body.get_quaternion()
        q = Quaternion(q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3])
        # cube local +Z axis in world:
        z_axis_world = np.array(q.rotate([0.0, 0.0, 1.0]))
        return float(np.dot(z_axis_world, np.array([0.0, 0.0, 1.0]))) >= float(self._UP_AXIS_COS_TOL)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # --- Per-cube checks: on table, upright-ish ---
        table_ok = True
        upright_ok = True

        positions = []
        for cube in self.cubes:
            pos = cube.body.get_position()
            positions.append(pos)

            # On the table top surface (collision is reliable in this direction)
            if not cube.is_colliding(self.table.table_top):
                table_ok = False

            # "Upright" check (cube is symmetric, but keep it explicit)
            if not self._is_upright(cube):
                upright_ok = False

        # --- Front-edge band (a range, not a precise line) + collinearity ---
        # The row may sit anywhere within the front band; the three cubes just need to
        # share roughly the same x so it reads as a single row along the front edge.
        xs = [float(p[0]) for p in positions]
        front_band_ok = all(self._FRONT_X_MIN <= x <= self._FRONT_X_MAX for x in xs)
        row_collinear_ok = (max(xs) - min(xs)) <= self._ROW_X_SPREAD_TOL
        x_ok = bool(front_band_ok and row_collinear_ok)

        # --- Row arrangement along Y at fixed X: pairwise Y separations ---
        ys = [float(p[1]) for p in positions]
        pairwise_seps = [abs(ys[i] - ys[j]) for i in range(3) for j in range(i + 1, 3)]
        y_sep_ok = all(self._MIN_PAIRWISE_Y_SEP <= s <= self._MAX_PAIRWISE_Y_SEP for s in pairwise_seps)

        # --- Non-overlap (extra safety): cubes should not be colliding with each other ---
        non_overlapping = True
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                if self.cubes[i].is_colliding(self.cubes[j]):
                    non_overlapping = False

        positions_ok = bool(x_ok and table_ok and upright_ok and y_sep_ok and non_overlapping)

        released = not any(
            self.robot.is_gripper_holding_object(obj, s) for obj in self.cubes for s in self.robot.grippers
        )

        # Mark all stages complete if final condition achieved
        if positions_ok and released:
            for i in range(self._NUM_SUBGOALS):
                self._metric_stage(i, True)

        self._success_check = bool(positions_ok and released)

        # Metrics: distance to the target line x=0.40 for each cube
        target_distance = {}
        pose_error = {}
        for name, cube in zip(self._cube_names, self.cubes):
            pos = cube.body.get_position()
            x_err = abs(float(pos[0]) - float(self._X_TARGET))
            target_distance[name] = float(x_err)
            pose_error[name] = float(x_err)

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})