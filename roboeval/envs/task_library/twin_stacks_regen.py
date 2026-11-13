# Standard imports that are ALWAYS prepended to task code
# This is the canonical source - validation agents should strip imports from LLM output

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
from roboeval.envs.props.items import ColoredCube, GoalPatch
from roboeval.envs.props.tables import Table
from roboeval.utils.env_utils import get_random_points_on_plane
from roboeval.utils.metric_rollout import MetricRolloutEval
from roboeval.utils.physics_utils import distance
from roboeval.utils.cube_factory import ensure_cubes
from roboeval.envs.props.semantic_cube import SemanticCube

class SideBySideCubeStacks(RoboEvalEnv, MetricRolloutEval):
    """
    Build two adjacent vertical stacks, each consisting of two colored cubes.
    The two stacks should be side-by-side (separated along the Y axis: left vs right)
    and roughly aligned in X (front/back).

    Sequential Sub-Goals (tracked as metric stages, not enforced as an action script):
        1. Form the left stack: cube_top_left on cube_bottom_left.
        2. Form the right stack: cube_top_right on cube_bottom_right.
        3. Place the two stack bases side-by-side in Y (left base at higher +Y).
        4. Align the two stack bases in X (front/back).
    """
    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Task geometry constants (ColoredCube is 4cm)
    _CUBE_SIZE = 0.04
    _BASE_Z = 0.97

    # Goal tolerances / constraints (from spec)
    _STACK_XY_TOL = 0.03
    _MIN_Z_CLEARANCE = 0.02
    _MIN_Y_SEP = 0.06
    _X_ALIGN_TOL = 0.03

    # Metric stages
    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Create props (Table already exists in preset)
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY asset allowed: ColoredCube)
        self.cube_bottom_left = ColoredCube(self._mojo)
        self.cube_top_left = ColoredCube(self._mojo)
        self.cube_bottom_right = ColoredCube(self._mojo)
        self.cube_top_right = ColoredCube(self._mojo)

        # Keep a consistent list for metrics and released-check
        self.cubes = [
            self.cube_bottom_left,
            self.cube_top_left,
            self.cube_bottom_right,
            self.cube_top_right,
        ]

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_bottom_left.set_color("red")
        self.cube_top_left.set_color("blue")
        self.cube_bottom_right.set_color("green")
        self.cube_top_right.set_color("yellow")

        # Define deterministic goal poses (within workspace bounds)
        # Axis convention: +Y left, -Y right; X is front/back.
        x0 = 0.55
        y_left = 0.12
        y_right = -0.12

        self._goal_poses = {
            "bottom_left": (np.array([x0, y_left, self._BASE_Z]), np.array([0.0, 0.0, 0.0, 1.0])),
            "top_left": (np.array([x0, y_left, self._BASE_Z + self._CUBE_SIZE]), np.array([0.0, 0.0, 0.0, 1.0])),
            "bottom_right": (np.array([x0, y_right, self._BASE_Z]), np.array([0.0, 0.0, 0.0, 1.0])),
            "top_right": (np.array([x0, y_right, self._BASE_Z + self._CUBE_SIZE]), np.array([0.0, 0.0, 0.0, 1.0])),
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
        # REQUIRED: Re-initialize metrics on every reset
        self._metric_init(
            track_vel_sync=True,
            track_vertical_sync=True,
            track_slippage=True,
            slip_objects=self.cubes,
            robot=self.robot,
            slip_sample_window=20,
            track_collisions=True,
        )

        # Deterministic RNG (required pattern)
        rng = np.random.default_rng(self.seed)

        # REQUIRED: set_color() in BOTH _initialize_env() AND _on_reset()
        self.cube_bottom_left.set_color("red")
        self.cube_top_left.set_color("blue")
        self.cube_bottom_right.set_color("green")
        self.cube_top_right.set_color("yellow")

        # Randomize initial poses in an UNSOLVED configuration (not stacked).
        # Workspace bounds: X [0.40, 0.70], Y [-0.25, 0.25], Z base = 0.97
        x_min, x_max = 0.40, 0.70
        y_min, y_max = -0.25, 0.25
        z0 = float(self._BASE_Z)

        def yaw_quat():
            yaw = float(rng.uniform(-np.pi, np.pi))
            q = Quaternion(axis=[0.0, 0.0, 1.0], angle=yaw)
            return np.array([q.x, q.y, q.z, q.w], dtype=float)

        # Sample 4 non-overlapping XY positions so cubes start separated on the table
        # (prevents accidental collisions/stacking during settle).
        min_xy_dist = 0.09  # > cube size to avoid contact after placement

        def sample_points(n: int):
            pts = []
            max_tries = 2000
            for _ in range(max_tries):
                x = float(rng.uniform(x_min, x_max))
                y = float(rng.uniform(y_min, y_max))
                p = np.array([x, y], dtype=float)
                if all(float(np.linalg.norm(p - q)) >= min_xy_dist for q in pts):
                    pts.append(p)
                    if len(pts) == n:
                        return pts
            # Fallback: fixed spread (still within bounds) if rejection sampling fails
            return [
                np.array([0.45, 0.20], dtype=float),
                np.array([0.65, 0.20], dtype=float),
                np.array([0.45, -0.20], dtype=float),
                np.array([0.65, -0.20], dtype=float),
            ]

        xy_pts = sample_points(4)

        # Place cubes one-by-one and forward to settle.
        cubes_in_order = [
            self.cube_bottom_left,
            self.cube_top_left,
            self.cube_bottom_right,
            self.cube_top_right,
        ]
        for cube, pxy in zip(cubes_in_order, xy_pts):
            pos = np.array([float(pxy[0]), float(pxy[1]), z0], dtype=float)
            cube.set_pose(position=pos, quat=yaw_quat())
            self._mojo.physics.forward()

        # Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    def _is_on(self, top: ColoredCube, bottom: ColoredCube, xy_tol: float, min_z_clearance: float) -> bool:
        """
        'on' relation: top is stacked on bottom.
        Uses collision as the primary signal plus XY alignment and Z clearance.
        """
        top_p = top.body.get_position()
        bot_p = bottom.body.get_position()

        xy_dist = float(np.linalg.norm((top_p - bot_p)[:2]))
        z_clear = float(top_p[2] - bot_p[2])

        colliding = top.is_colliding(bottom)
        not_on_table = not top.is_colliding(self.table.table_top)

        # Also keep stacking height reasonable (avoid false positives from far-away collisions)
        near_expected_height = abs(z_clear - self._CUBE_SIZE) <= 0.03  # forgiving

        return bool(
            colliding
            and not_on_table
            and (xy_dist <= xy_tol)
            and (z_clear >= min_z_clearance)
            and near_expected_height
        )

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Goal checks
        left_stack_ok = self._is_on(
            self.cube_top_left, self.cube_bottom_left,
            xy_tol=self._STACK_XY_TOL, min_z_clearance=self._MIN_Z_CLEARANCE
        )
        right_stack_ok = self._is_on(
            self.cube_top_right, self.cube_bottom_right,
            xy_tol=self._STACK_XY_TOL, min_z_clearance=self._MIN_Z_CLEARANCE
        )

        bl = self.cube_bottom_left.body.get_position()
        br = self.cube_bottom_right.body.get_position()

        # Left (+Y) base must be left of right (-Y) base by at least min separation
        y_sep = float(bl[1] - br[1])
        side_by_side_ok = y_sep >= self._MIN_Y_SEP

        # Bases aligned in X (front/back)
        x_align_err = float(abs(bl[0] - br[0]))
        x_aligned_ok = x_align_err <= self._X_ALIGN_TOL

        positions_ok = bool(left_stack_ok and right_stack_ok and side_by_side_ok and x_aligned_ok)
        semantic_ok = True  # no semantic constraints beyond color assignment (not part of success)

        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Metric subgoals (stages)
        self._metric_stage(0, bool(left_stack_ok))
        self._metric_stage(1, bool(right_stack_ok))
        self._metric_stage(2, bool(side_by_side_ok))
        self._metric_stage(3, bool(x_aligned_ok))

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Metrics: distances to goal and simple constraint errors
        def pose_pos_err(cube: ColoredCube, goal_key: str) -> float:
            goal_pos, _ = self._goal_poses[goal_key]
            return float(np.linalg.norm(cube.body.get_position() - goal_pos))

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance={
                "xy_left_stack": float(np.linalg.norm((self.cube_top_left.body.get_position() - self.cube_bottom_left.body.get_position())[:2])),
                "xy_right_stack": float(np.linalg.norm((self.cube_top_right.body.get_position() - self.cube_bottom_right.body.get_position())[:2])),
                "x_base_align_err": x_align_err,
                "y_base_sep": y_sep,
            },
            pose_error={
                "cube_bottom_left_pos_err": pose_pos_err(self.cube_bottom_left, "bottom_left"),
                "cube_top_left_pos_err": pose_pos_err(self.cube_top_left, "top_left"),
                "cube_bottom_right_pos_err": pose_pos_err(self.cube_bottom_right, "bottom_right"),
                "cube_top_right_pos_err": pose_pos_err(self.cube_top_right, "top_right"),
            },
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})