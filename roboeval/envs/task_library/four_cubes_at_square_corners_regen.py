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

class ColoredCubeSquarePlacement(RoboEvalEnv, MetricRolloutEval):
    """
    Arrange four colored cubes on the table so they occupy the four corners of an
    imaginary 12 cm by 12 cm square aligned with the workspace X/Y axes.

    Axis convention:
      +X = back, -X = front
      +Y = left, -Y = right
      +Z = up

    Goal corner assignment:
      - cube1: back-right corner (reference)
      - cube2: back-left  (left (+Y) of cube1 by 0.12 m)
      - cube4: front-right (front (-X) of cube1 by 0.12 m)
      - cube3: front-left (left of cube4 AND front of cube2 by 0.12 m)

    Sequential Sub-Goals:
        1. Make cube2 left (+Y) of cube1 by 12 cm.
        2. Make cube4 front (-X) of cube1 by 12 cm.
        3. Make cube3 left (+Y) of cube4 by 12 cm.
        4. Make cube3 front (-X) of cube2 by 12 cm.
        5. Keep all cubes coplanar on the tabletop (no stacking).
        6. Release all cubes (no cube held in any gripper).
    """

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    _SIDE_LEN_M = 0.12
    _REL_TOL_M = 0.015
    _TABLE_Z_TOL_M = 0.01
    _CUBE_REST_Z = 0.97

    _NUM_SUBGOALS = 6

    def _initialize_env(self):
        # Props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create objects (ONLY ColoredCube allowed)
        self.cube1 = ColoredCube(self._mojo)
        self.cube2 = ColoredCube(self._mojo)
        self.cube3 = ColoredCube(self._mojo)
        self.cube4 = ColoredCube(self._mojo)
        self.cubes = [self.cube1, self.cube2, self.cube3, self.cube4]

        # REQUIRED: set_color() in _initialize_env
        self.cube1.set_color("red")
        self.cube2.set_color("blue")
        self.cube3.set_color("green")
        self.cube4.set_color("yellow")

        # Define a canonical goal square inside workspace bounds
        # cube1 is back-right reference; square spans +Y (left) and -X (front)
        p1 = np.array([0.60, -0.06, self._CUBE_REST_Z])  # cube1 (back-right)
        p2 = np.array([0.60, +0.06, self._CUBE_REST_Z])  # cube2 (back-left)
        p4 = np.array([0.48, -0.06, self._CUBE_REST_Z])  # cube4 (front-right)
        p3 = np.array([0.48, +0.06, self._CUBE_REST_Z])  # cube3 (front-left)

        self._goal_positions = {
            "cube1": p1,
            "cube2": p2,
            "cube3": p3,
            "cube4": p4,
        }

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

        # 2. Deterministic RNG (required pattern; validation may add randomization later)
        rng = np.random.default_rng(self.seed)

        # 3. Place objects in RANDOM (unsolved) positions
        # REQUIRED: set_color() in _on_reset
        self.cube1.set_color("red")
        self.cube2.set_color("blue")
        self.cube3.set_color("green")
        self.cube4.set_color("yellow")

        x_lo, x_hi = 0.40, 0.70
        y_lo, y_hi = -0.25, 0.25
        z = self._CUBE_REST_Z

        # Keep cubes reasonably separated to avoid overlap/collisions.
        min_xy_sep = 0.06

        def sample_pose():
            x = float(rng.uniform(x_lo, x_hi))
            y = float(rng.uniform(y_lo, y_hi))
            # Random yaw about Z
            yaw = float(rng.uniform(-np.pi, np.pi))
            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            quat = np.array([0.0, 0.0, qz, qw], dtype=float)
            return np.array([x, y, z], dtype=float), quat

        def far_enough(p, ps):
            for pp in ps:
                if np.linalg.norm(p[:2] - pp[:2]) < min_xy_sep:
                    return False
            return True

        # Rejection sample until the relational goal is NOT satisfied (guarantee _success() False post-reset).
        max_tries = 5000
        positions = None
        quats = None
        for _ in range(max_tries):
            ps = []
            qs = []
            for _k in range(4):
                p, q = sample_pose()
                if not far_enough(p, ps):
                    break
                ps.append(p)
                qs.append(q)
            if len(ps) != 4:
                continue

            p1, p2, p3, p4 = ps[0], ps[1], ps[2], ps[3]

            g1 = self._axis_relative_ok(p2, p1, axis="y", signed_delta_m=+self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
            g2 = self._axis_relative_ok(p4, p1, axis="x", signed_delta_m=-self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
            g3 = self._axis_relative_ok(p3, p4, axis="y", signed_delta_m=+self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
            g4 = self._axis_relative_ok(p3, p2, axis="x", signed_delta_m=-self._SIDE_LEN_M, tol_m=self._REL_TOL_M)

            # Ensure the square constraints are NOT all simultaneously satisfied.
            if not (g1 and g2 and g3 and g4):
                positions, quats = ps, qs
                break

        if positions is None:
            # Fallback: deterministic "obviously unsolved" layout well within bounds
            positions = [
                np.array([0.44, -0.20, z], dtype=float),
                np.array([0.68, +0.22, z], dtype=float),
                np.array([0.41, +0.18, z], dtype=float),
                np.array([0.69, -0.22, z], dtype=float),
            ]
            quats = [np.array([0.0, 0.0, 0.0, 1.0], dtype=float)] * 4

        self.cube1.set_pose(position=positions[0], quat=quats[0])
        self.cube2.set_pose(position=positions[1], quat=quats[1])
        self.cube3.set_pose(position=positions[2], quat=quats[2])
        self.cube4.set_pose(position=positions[3], quat=quats[3])

        # 4. Physics settling
        self._mojo.physics.forward()

        # 5. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _axis_relative_ok(
        self,
        subject_pos: np.ndarray,
        target_pos: np.ndarray,
        axis: str,
        signed_delta_m: float,
        tol_m: float,
    ) -> bool:
        ax = {"x": 0, "y": 1, "z": 2}[axis]
        delta = float(subject_pos[ax] - target_pos[ax])
        return abs(delta - signed_delta_m) <= tol_m

    def _on_table_and_coplanar(self) -> bool:
        # All cubes should be resting on the table (not stacked) and near rest Z.
        for c in self.cubes:
            p = c.body.get_position()
            if abs(float(p[2]) - self._CUBE_REST_Z) > self._TABLE_Z_TOL_M:
                return False
            # Use table surface collision (reliable)
            if not c.is_colliding(self.table.table_top):
                return False

        # No cube-cube collisions (helps enforce "no stacking"/no overlap)
        for i in range(len(self.cubes)):
            for j in range(i + 1, len(self.cubes)):
                if self.cubes[i].is_colliding(self.cubes[j]):
                    return False
        return True

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table") or not hasattr(self, "_goal_positions"):
            return False

        p1 = self.cube1.body.get_position()
        p2 = self.cube2.body.get_position()
        p3 = self.cube3.body.get_position()
        p4 = self.cube4.body.get_position()

        # Goal conditions (signed deltas based on axis convention)
        # cube2 left_of cube1: +Y by 0.12
        g1 = self._axis_relative_ok(p2, p1, axis="y", signed_delta_m=+self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
        # cube4 front_of cube1: -X by 0.12
        g2 = self._axis_relative_ok(p4, p1, axis="x", signed_delta_m=-self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
        # cube3 left_of cube4: +Y by 0.12
        g3 = self._axis_relative_ok(p3, p4, axis="y", signed_delta_m=+self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
        # cube3 front_of cube2: -X by 0.12
        g4 = self._axis_relative_ok(p3, p2, axis="x", signed_delta_m=-self._SIDE_LEN_M, tol_m=self._REL_TOL_M)
        # all cubes coplanar on table
        g5 = self._on_table_and_coplanar()

        positions_ok = bool(g1 and g2 and g3 and g4 and g5)
        semantic_ok = True  # no semantic constraints beyond having colors set

        released = not any(
            self.robot.is_gripper_holding_object(obj, g)
            for obj in self.cubes
            for g in self.robot.grippers
        )

        # Subgoal staging
        self._metric_stage(0, bool(g1))
        self._metric_stage(1, bool(g2))
        self._metric_stage(2, bool(g3))
        self._metric_stage(3, bool(g4))
        self._metric_stage(4, bool(g5))
        self._metric_stage(5, bool(released))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: relation errors + pose errors to canonical goal
        rel_err = {
            "cube2_left_of_cube1_y_err_m": abs(float((p2[1] - p1[1]) - self._SIDE_LEN_M)),
            "cube4_front_of_cube1_x_err_m": abs(float((p4[0] - p1[0]) + self._SIDE_LEN_M)),
            "cube3_left_of_cube4_y_err_m": abs(float((p3[1] - p4[1]) - self._SIDE_LEN_M)),
            "cube3_front_of_cube2_x_err_m": abs(float((p3[0] - p2[0]) + self._SIDE_LEN_M)),
        }

        pose_err = {
            "cube1_goal_dist_m": float(np.linalg.norm(p1 - self._goal_positions["cube1"])),
            "cube2_goal_dist_m": float(np.linalg.norm(p2 - self._goal_positions["cube2"])),
            "cube3_goal_dist_m": float(np.linalg.norm(p3 - self._goal_positions["cube3"])),
            "cube4_goal_dist_m": float(np.linalg.norm(p4 - self._goal_positions["cube4"])),
        }

        # Finalize metrics at END (MUST be last metric call before return)
        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=rel_err,
            pose_error=pose_err,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})