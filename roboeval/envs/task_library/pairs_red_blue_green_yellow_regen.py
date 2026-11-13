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

class AdjacentColorCubePairing(RoboEvalEnv, MetricRolloutEval):
    """Make two adjacent color pairs: red-blue on the left and green-yellow on the right.

Success Criteria:
- Red and blue form an adjacent pair in the left-side region; green and yellow form an adjacent pair in the right-side region.
- All cubes remain in workspace bounds, on the table, and the two pairs are separated by the minimum pair distance.
- No gripper is holding any cube."""

    _PRESET_PATH = PRESETS_PATH / "bimanual_setup.yaml"
    USE_TEMPORAL_SUCCESS = False
    _success_check = True
    _final_metrics: Dict = {}

    # Workspace (task spec bounds)
    _X_MIN, _X_MAX = 0.40, 0.70
    _Y_MIN, _Y_MAX = -0.25, 0.25

    # Region constraints (task spec)
    # A 10 cm dead band between the halves punished pairs that were plainly on one
    # side; narrow it, and let the region edges run past the sampling bounds so a
    # cube resting just outside does not fail on a technicality.
    _LEFT_Y_MIN, _LEFT_Y_MAX = 0.03, 0.30     # +Y is left
    _RIGHT_Y_MIN, _RIGHT_Y_MAX = -0.30, -0.03  # -Y is right
    _BOUNDS_MARGIN_M = 0.03

    # Geometry / tolerances
    _CUBE_SIZE_M = 0.04
    # Edge gap is a center-distance approximation that assumes axis-aligned cubes.
    # A tilted pair reads a larger center distance even when the faces touch, and
    # operators reported correct-looking pairs failing for exactly that reason, so
    # the gap allowance covers a 45-degree tilt (0.04*sqrt(2) - 0.04 ~= 0.017) with
    # room to spare.
    _MAX_EDGE_GAP_XY_M = 0.05
    _MAX_Z_DIFF_M = 0.025
    _MIN_PAIR_SEPARATION_XY_M = 0.08
    _MAX_HEIGHT_ABOVE_TABLE_M = 0.03

    _NUM_SUBGOALS = 4

    def _initialize_env(self):
        # Table already exists in preset
        self.table = self._preset.get_props(Table)[0]

        # Create cubes (ONLY allowed asset)
        self.cube_red = ColoredCube(self._mojo)
        self.cube_blue = ColoredCube(self._mojo)
        self.cube_green = ColoredCube(self._mojo)
        self.cube_yellow = ColoredCube(self._mojo)

        # REQUIRED: set_color() in _initialize_env()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_green.set_color("green")
        self.cube_yellow.set_color("yellow")

        self.cubes = [self.cube_red, self.cube_blue, self.cube_green, self.cube_yellow]

        # Deterministic "goal" poses used for pose_error / target_distance metrics
        # (Pairs are adjacent by center distance ~= cube size, and separated across Y.)
        self._goal_poses = {
            "cube_red": np.array([0.55, 0.16, 0.97]),
            "cube_blue": np.array([0.55, 0.12, 0.97]),
            "cube_green": np.array([0.55, -0.16, 0.97]),
            "cube_yellow": np.array([0.55, -0.12, 0.97]),
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

        # 2. Deterministic RNG (pipeline may override later)
        rng = np.random.default_rng(self.seed)

        # 3. REQUIRED: set_color() in _on_reset()
        self.cube_red.set_color("red")
        self.cube_blue.set_color("blue")
        self.cube_green.set_color("green")
        self.cube_yellow.set_color("yellow")

        # 4. Place objects in RANDOM *UNSOLVED* positions within workspace bounds.
        #    We guarantee _success() is False by placing all cubes near Y≈0 (outside both left/right regions)
        #    AND by spacing each intended pair far enough so they are not adjacent.
        x0 = float(rng.uniform(self._X_MIN + 0.06, self._X_MAX - 0.06))
        y0 = float(rng.uniform(-0.03, 0.03))  # outside both regions: (-0.05, 0.05)
        z = 0.97

        # Fixed offsets (with slight random jitter) to ensure non-adjacency and bounds
        jx = float(rng.uniform(-0.005, 0.005))
        jy = float(rng.uniform(-0.005, 0.005))

        pr = np.array([x0 - 0.06 + jx, y0 + 0.02 + jy, z], dtype=float)
        pb = np.array([x0 + 0.06 - jx, y0 - 0.02 - jy, z], dtype=float)
        pg = np.array([x0 - 0.06 - jx, y0 - 0.02 + jy, z], dtype=float)
        py = np.array([x0 + 0.06 + jx, y0 + 0.02 - jy, z], dtype=float)

        # Random yaw for each cube (kept upright)
        def yaw_quat() -> np.ndarray:
            yaw = float(rng.uniform(-np.pi, np.pi))
            # set_pose reads [w, x, y, z]; xyzw made this a 180 deg flip, not a yaw.
            return Quaternion(axis=[0.0, 0.0, 1.0], radians=yaw).elements

        self.cube_red.set_pose(position=pr, quat=yaw_quat())
        self.cube_blue.set_pose(position=pb, quat=yaw_quat())
        self.cube_green.set_pose(position=pg, quat=yaw_quat())
        self.cube_yellow.set_pose(position=py, quat=yaw_quat())

        # 5. Physics settling
        self._mojo.physics.forward()

        # 6. Initialize subgoals to False
        for idx in range(self._NUM_SUBGOALS):
            self._metric_stage(idx, False)

    def _on_step(self):
        if self.USE_TEMPORAL_SUCCESS:
            super()._on_step()
        self._metric_step()

    @staticmethod
    def _xy_dist(a: np.ndarray, b: np.ndarray) -> float:
        d = a[:2] - b[:2]
        return float(np.linalg.norm(d))

    def _edge_gap_xy(self, a_pos: np.ndarray, b_pos: np.ndarray) -> float:
        """Approximate edge-to-edge gap in XY assuming 4cm cubes."""
        center_dist = self._xy_dist(a_pos, b_pos)
        return float(center_dist - self._CUBE_SIZE_M)

    def _in_bounds(self, pos: np.ndarray) -> bool:
        m = self._BOUNDS_MARGIN_M
        return (
            (self._X_MIN - m <= pos[0] <= self._X_MAX + m)
            and (self._Y_MIN - m <= pos[1] <= self._Y_MAX + m)
        )

    @staticmethod
    def _in_y_range(pos: np.ndarray, y_min: float, y_max: float) -> bool:
        return bool(y_min <= pos[1] <= y_max)

    def _on_table(self, cube: ColoredCube) -> bool:
        pos = cube.body.get_position()
        height_ok = (pos[2] <= (0.97 + self._MAX_HEIGHT_ABOVE_TABLE_M))
        # Use table.table_top (reliable collision geom) when available
        colliding_table = cube.is_colliding(self.table.table_top)
        return bool(height_ok and colliding_table)

    def _adjacent_pair_ok(self, a: ColoredCube, b: ColoredCube) -> bool:
        pa = a.body.get_position()
        pb = b.body.get_position()
        edge_gap_ok = (self._edge_gap_xy(pa, pb) <= self._MAX_EDGE_GAP_XY_M)
        z_diff_ok = (abs(float(pa[2] - pb[2])) <= self._MAX_Z_DIFF_M)
        return bool(edge_gap_ok and z_diff_ok)

    def _success(self) -> bool:
        # ⚠️ CRITICAL: RESET success flag at START
        self._success_check = False

        # Defensive checks - _success() is called during reset!
        if not hasattr(self, "cubes") or not hasattr(self, "table"):
            return False

        # Gather positions
        pr = self.cube_red.body.get_position()
        pb = self.cube_blue.body.get_position()
        pg = self.cube_green.body.get_position()
        py = self.cube_yellow.body.get_position()

        # Workspace bounds (task spec)
        bounds_ok = all(self._in_bounds(p) for p in [pr, pb, pg, py])

        # Pair adjacency checks
        left_adj_ok = self._adjacent_pair_ok(self.cube_red, self.cube_blue)
        right_adj_ok = self._adjacent_pair_ok(self.cube_green, self.cube_yellow)

        # Region checks (+Y = left, -Y = right)
        left_region_ok = (
            self._in_y_range(pr, self._LEFT_Y_MIN, self._LEFT_Y_MAX)
            and self._in_y_range(pb, self._LEFT_Y_MIN, self._LEFT_Y_MAX)
        )
        right_region_ok = (
            self._in_y_range(pg, self._RIGHT_Y_MIN, self._RIGHT_Y_MAX)
            and self._in_y_range(py, self._RIGHT_Y_MIN, self._RIGHT_Y_MAX)
        )

        # Distinct pairs: separate left vs right (use red-green, per spec)
        sep_ok = (self._xy_dist(pr, pg) >= self._MIN_PAIR_SEPARATION_XY_M)

        # On-table checks
        on_table_ok = all(self._on_table(c) for c in self.cubes)

        # Released gripper (no grasp-state goals; but ensure not being held)
        released = not any(
            self.robot.is_gripper_holding_object(obj, s)
            for obj in self.cubes
            for s in self.robot.grippers
        )

        # Stage metrics (sequential, single-arm-friendly)
        if left_adj_ok:
            self._metric_stage(0, True)
        if left_adj_ok and left_region_ok:
            self._metric_stage(1, True)
        if right_adj_ok:
            self._metric_stage(2, True)
        if right_adj_ok and right_region_ok and sep_ok:
            self._metric_stage(3, True)

        positions_ok = bool(
            bounds_ok
            and left_adj_ok
            and right_adj_ok
            and left_region_ok
            and right_region_ok
            and sep_ok
            and on_table_ok
        )

        semantic_ok = True  # Colors are fixed by construction (but not a success condition to re-verify here)

        self._success_check = bool(positions_ok and semantic_ok and released)

        # Finalize metrics (MUST be last metric call before return)
        def _pose_err(cube: ColoredCube, goal: np.ndarray) -> float:
            p = cube.body.get_position()
            return float(np.linalg.norm(p - goal))

        target_distance = {
            "edge_gap_left_pair_xy_m": max(0.0, self._edge_gap_xy(pr, pb)),
            "edge_gap_right_pair_xy_m": max(0.0, self._edge_gap_xy(pg, py)),
            "pair_separation_red_green_xy_m": self._xy_dist(pr, pg),
        }
        pose_error = {
            "cube_red_to_goal_m": _pose_err(self.cube_red, self._goal_poses["cube_red"]),
            "cube_blue_to_goal_m": _pose_err(self.cube_blue, self._goal_poses["cube_blue"]),
            "cube_green_to_goal_m": _pose_err(self.cube_green, self._goal_poses["cube_green"]),
            "cube_yellow_to_goal_m": _pose_err(self.cube_yellow, self._goal_poses["cube_yellow"]),
        }

        self._final_metrics = self._metric_finalize(
            success_flag=self._success_check,
            target_distance=target_distance,
            pose_error=pose_error,
        )
        return self._success_check

    def _get_task_info(self):
        return getattr(self, "_final_metrics", {})